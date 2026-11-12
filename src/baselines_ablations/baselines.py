from types import SimpleNamespace
import copy,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
from common import *
from protocol import load,loader

def recipe(dataset,model,horizon):
 if model=='GTR':return json.loads((ROOT/'audit/gtr_recipes.json').read_text())[dataset]|{'pred_len':horizon}
 if dataset=='PEMS08':
  lock=ROOT/'audit/recipe_locks'/f'{model}_h{horizon}.json'
  if not lock.exists():raise RuntimeError('PEMS08 validation-only recipe lock required')
  return json.loads(lock.read_text())['recipe']
 source_ds='PEMS04' if dataset=='PEMS07' else dataset
 rr=[r for r in old_records() if r['dataset']==source_ds and r['model']==model and r['horizon']==horizon and r['seed']==2024 and r['completed_test']]
 assert len(rr)==1,(dataset,model,horizon,len(rr))
 a=recipe_fields(rr[0]['args']);a['enc_in']=SPECS[dataset]['nodes'];a['cycle']=SPECS[dataset]['cycle']
 overrides=ROOT/'audit/batch_overrides.json'
 if overrides.exists():a['batch_size']=json.loads(overrides.read_text()).get(dataset+'_'+model,a['batch_size'])
 return a

def build(model,args):
 a=SimpleNamespace(**args)
 if model=='GTR':
  p=ROOT/'vendor/GTR/models/GTR.py';s=importlib.util.spec_from_file_location('_official_gtr',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 elif model=='PatchTST':from models import PatchTST as m
 elif model=='iTransformer':from models import iTransformer as m
 else:raise ValueError(model)
 return m.Model(a).float()

def forward(model,name,x,origins,marks,cycle):
 if name=='GTR':return model(x,origins.remainder(cycle))
 if name=='iTransformer':return model(x,marks,None,None)
 return model(x)

@torch.no_grad()
def evaluate(model,name,data,h,batch,device,split,cycle,legacy_selection=False):
 model.eval();se=ae=0.;count=windows=0;losses=[]
 dl=loader(data,h,split,batch,shuffle=legacy_selection,drop_last=legacy_selection)
 for x,y,o,marks in dl:
  x=x.to(device);y=y.to(device);o=o.to(device);marks=marks.to(device)
  pred=forward(model,name,x,o,marks,cycle)
  if pred.shape!=y.shape or not torch.isfinite(pred).all():raise RuntimeError('invalid baseline output')
  error=pred-y;se+=float(error.square().sum(dtype=torch.float64));ae+=float(error.abs().sum(dtype=torch.float64));count+=y.numel();windows+=len(x)
  losses.append(float(torch.mean(error.square()).cpu()))
 if not count:raise RuntimeError('empty evaluation')
 if not legacy_selection:assert windows==len(data.origins(h,split))
 return dict(mse=se/count,mae=ae/count,windows=windows,selection_loss=float(np.mean(losses)))

def select_recipe(job,device):
 ds='PEMS08';name=job['model'];h=job['horizon'];data=load(ds)
 path=ROOT/'audit/recipe_locks'/f'{name}_h{h}.json'
 if path.exists():return json.loads(path.read_text())
 candidates=[r for r in old_records() if r['dataset']==ds and r['model']==name and r['seed']==2024 and r['horizon']==h and r['completed_test']]
 scored=[]
 for r in sorted(candidates,key=lambda r:r['path']):
  a=recipe_fields(r['args']);ck=checkpoint_for_record(r);before=sha(ck);seed_all(2024);m=build(name,a).to(device)
  payload=torch.load(ck,map_location='cpu',weights_only=False);m.load_state_dict(payload.get('model',payload) if isinstance(payload,dict) else payload,strict=True)
  # Full identical validation origins. The selector never reads stored test scores.
  vm=evaluate(m,name,data,h,min(a['batch_size'],16),device,'val',a['cycle'])
  assert sha(ck)==before
  scored.append(dict(recipe=a,source_log=r['path'],checkpoint=str(ck),checkpoint_sha256=before,validation=vm))
  del m;torch.cuda.empty_cache()
 best=min(scored,key=lambda r:(r['validation']['mse'],r['source_log']))
 out=dict(dataset=ds,model=name,horizon=h,selection_split='validation',test_labels_used_for_selection=False,recipe=best['recipe'],candidates=scored,chosen_source_log=best['source_log'],data_fingerprint=data.metadata['fingerprint'])
 write_json(path,out);log('RECIPE_LOCK',model=name,horizon=h,source=best['source_log'],validation_mse=best['validation']['mse'])
 return out

def compatible_reuse(ds,name,h,seed,a):
 matches=[r for r in old_records() if r['dataset']==ds and r['model']==name and r['horizon']==h and r['seed']==seed and r['completed_test'] and recipe_fields(r['args'])==a]
 return sorted(matches,key=lambda r:r['path'])[0] if matches else None

def run(job,device):
 start=time.monotonic();ds=job['dataset'];name=job['model'];h=job['horizon'];seed=job['seed'];a=recipe(ds,name,h);data=load(ds)
 source=compatible_reuse(ds,name,h,seed,a) if name!='GTR' else None
 outpath=result_path(job)
 if outpath.exists():return
 if source:
  checkpoint=checkpoint_for_record(source)
  out=dict(job=job,model=name,dataset=ds,horizon=h,seed=seed,status='COMPLETE',reused=True,recipe=a,protocol=data.metadata,
           test=source['metrics'][-1],test_windows=len(data.origins(h,'test')),source_log=source['path'],source_log_sha256=source['sha256'],checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),test_labels_used_for_selection=False)
  write_json(outpath,out);log('BASELINE_REUSED',job=job['id']);return
 seed_all(seed);model=build(name,a).to(device);opt=torch.optim.Adam(model.parameters(),lr=a['learning_rate'])
 train=loader(data,h,'train',a['batch_size'],shuffle=True,drop_last=True)
 scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a['learning_rate'],epochs=a['train_epochs'],steps_per_epoch=len(train),pct_start=a.get('pct_start',.3))
 best=float('inf');bad=0;history=[];ck=outpath.parent/'model.pt';ck.parent.mkdir(parents=True,exist_ok=True)
 torch.cuda.reset_peak_memory_stats(device)
 for epoch in range(1,a['train_epochs']+1):
  model.train();total=0.;steps=0;tick=time.monotonic()
  for i,(x,y,o,marks) in enumerate(train):
   x=x.to(device);y=y.to(device);o=o.to(device);marks=marks.to(device);opt.zero_grad(set_to_none=True)
   pred=forward(model,name,x,o,marks,a['cycle']);loss=torch.mean((pred-y).square())
   if not torch.isfinite(loss):raise RuntimeError('nonfinite baseline objective')
   loss.backward();opt.step();total+=float(loss.detach());steps+=1
   if a.get('lradj')=='TST':scheduler.step()
   if time.monotonic()-tick>45:log('TRAIN_PROGRESS',job=job['id'],epoch=epoch,batch=i+1,batches=len(train));tick=time.monotonic()
  vm=evaluate(model,name,data,h,a['batch_size'],device,'val',a['cycle'],legacy_selection=True)
  # Legacy trainers iterated a non-shuffled test loader every epoch. Reproduce
  # only its CPU base-seed draw, without consulting any test inputs or labels.
  torch.empty((),dtype=torch.int64).random_()
  row=dict(epoch=epoch,train_mse=total/max(steps,1),validation=vm);history.append(row);log('BASELINE_EPOCH',job=job['id'],**row)
  if vm['selection_loss']<best:
   best=vm['selection_loss'];bad=0;torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},recipe=a,epoch=epoch,validation=vm),ck)
  else:bad+=1
  if bad>=a['patience']:break
  if a.get('lradj','type3')=='type3':lr=a['learning_rate'] if epoch<3 else a['learning_rate']*.8**(epoch-3)
  elif a['lradj']=='type1':lr=a['learning_rate']*.5**(epoch-1)
  elif a['lradj']=='cosine':lr=a['learning_rate']/10*(1+np.cos(epoch/a['train_epochs']*np.pi))
  elif a['lradj']=='TST':lr=opt.param_groups[0]['lr']
  else:raise ValueError('unsupported schedule '+a['lradj'])
  for group in opt.param_groups:group['lr']=float(lr)
 state=torch.load(ck,map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True)
 vm=evaluate(model,name,data,h,a['batch_size'],device,'val',a['cycle'])
 release=dict(checkpoint_sha256=sha(ck),recipe_sha256=digest(a),selected_epoch=state['epoch'],validation=vm,test_labels_used_for_selection=False)
 write_json(ck.parent/'test_release.json',release)
 tm=evaluate(model,name,data,h,a['batch_size'],device,'test',a['cycle'])
 assert sha(ck)==release['checkpoint_sha256']
 out=dict(job=job,model=name,dataset=ds,horizon=h,seed=seed,status='COMPLETE',reused=False,recipe=a,protocol=data.metadata,test=tm,validation=vm,best_epoch=state['epoch'],history=history,
          checkpoint=str(ck),checkpoint_sha256=release['checkpoint_sha256'],test_labels_used_for_selection=False,
          parameters=sum(p.numel() for p in model.parameters()),peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,seconds=time.monotonic()-start)
 write_json(outpath,out);log('BASELINE_RESULT',job=job['id'],test=tm,seconds=out['seconds'])
