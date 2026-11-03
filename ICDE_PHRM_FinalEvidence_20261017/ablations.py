"""Frozen retrieval caches accelerate training without changing trainable math."""
import argparse,json,shutil,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common import *
from protocol import load,loader
from models.tqnet import Model as TQNet
from ablation_model import Adapter,graph_for,build_bank,VARIANTS
import frozen_physics as frozen
from structure_controls import graph_for as structure_graph, VARIANTS as STRUCTURE_VARIANTS, memory_storage
from ablation_model import graph_for as physical_graph, VARIANTS as MODEL_VARIANTS
VARIANTS=MODEL_VARIANTS+STRUCTURE_VARIANTS
def graph_for(ds,variant):
 return structure_graph(ds,variant) if variant in STRUCTURE_VARIANTS else physical_graph(ds,variant)

def base_path(ds,h,seed):
 entry=json.loads((ROOT/'audit/checkpoint_map.json').read_text())[f'{ds}_h{h}_s{seed}']
 path=Path(entry['checkpoint'])
 if sha(path)!=entry['sha256']:raise RuntimeError('paired checkpoint changed')
 return path

def setup(job,device):
 ds=job['dataset'];h=job['horizon'];seed=job['seed'];v=job['variant'];data=load(ds);seed_all(seed)
 # Preserve initialization order of the corresponding frozen production trainer.
 daily,weekly=frozen.periodic_prior(np.array(data.values[:data.metadata['train_end']],copy=True),288,2016);daily=daily.to(device);weekly=weekly.to(device)
 graph=graph_for(ds,v)
 if ds=='PEMS07':
  # The graph-repair trainer constructs the graph before the unchanged base.
  pass
 cfg=dict(seq_len=96,pred_len=h,enc_in=SPECS[ds]['nodes'],cycle=288,model_type='mlp',d_model=512,dropout=0,use_revin=int(ds=='PEMS08'))
 base=TQNet(SimpleNamespace(**cfg)).float();ck=base_path(ds,h,seed);payload=torch.load(ck,map_location='cpu',weights_only=False);base.load_state_dict(payload.get('model',payload),strict=True);base=base.to(device).eval()
 if v=='raw_memory':daily=torch.zeros_like(daily);weekly=torch.zeros_like(weekly)
 model=Adapter(base,graph,daily,weekly,h,v).to(device)
 stride=job.get('memory_stride',12);bank_o=data.origins(h,'train')[::stride]
 if len(bank_o)==0 or bank_o[-1]+h>data.metadata['train_end']:raise RuntimeError('unsafe memory endpoint')
 bank=None
 if v not in ('periodic_only','residual_mlp'):
  bank=build_bank(data,bank_o,h,graph,daily,weekly,device,32 if ds=='PEMS08' else 4,v,job.get('topk',8))
 return data,model,bank,graph,ck,bank_o

class Cached(Dataset):
 def __init__(self,directory,data,origins,h):
  self.arrays={k:np.load(directory/(k+'.npy'),mmap_mode='r') for k in ('base','diff','features')};self.data=data;self.origins=origins;self.h=h
 def __len__(self):return len(self.origins)
 def __getitem__(self,i):
  t=int(self.origins[i]);return *(torch.from_numpy(self.arrays[k][i].copy()) for k in ('base','diff','features')),torch.from_numpy(self.data.values[t:t+self.h].copy())

@torch.no_grad()
def create_cache(path,data,origins,h,model,bank,batch,device,job_id):
 path.mkdir(parents=True,exist_ok=True);n=len(origins);nodes=data.values.shape[1]
 arrays={k:np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype='float32',shape=shape) for k,shape in [('base',(n,h,nodes)),('diff',(n,h,nodes)),('features',(n,nodes,6))]}
 model.eval();last=time.monotonic()
 for start in range(0,n,batch):
  oo=origins[start:start+batch];o=torch.tensor(oo,device=device);x=torch.from_numpy(np.stack([data.values[t-96:t] for t in oo])).to(device)
  base,diff,features=model.decompose(x,o,bank)
  for k,a in zip(('base','diff','features'),(base,diff,features)):arrays[k][start:start+len(oo)]=a.cpu().numpy()
  if time.monotonic()-last>35:log('CACHE_PROGRESS',job=job_id,done=start+len(oo),total=n);last=time.monotonic()
 for a in arrays.values():a.flush()
 del arrays

def cached_loader(directory,data,origins,h,batch,shuffle):return DataLoader(Cached(directory,data,origins,h),batch_size=batch,shuffle=shuffle,num_workers=2,pin_memory=True)

@torch.no_grad()
def evaluate_cached(model,dl,device,weight=None,calibrate=False):
 model.eval();se=ae=bse=bae=0.;count=0;h=model.horizon;grid=torch.linspace(0,1,9,device=device)
 scores=torch.zeros(h,9,dtype=torch.float64,device=device) if calibrate else None;abscores=torch.zeros_like(scores) if calibrate else None;perstepcount=0
 for b,d,f,y in dl:
  b=b.to(device);d=d.to(device);f=f.to(device);y=y.to(device);pred,delta=model.mix(b,d,f,weight);e=pred-y;be=b-y
  se+=float(e.square().sum(dtype=torch.float64));ae+=float(e.abs().sum(dtype=torch.float64));bse+=float(be.square().sum(dtype=torch.float64));bae+=float(be.abs().sum(dtype=torch.float64));count+=y.numel()
  if calibrate:
   for j,w in enumerate(grid):
    ee=be+w*delta;scores[:,j]+=ee.square().sum((0,2),dtype=torch.float64);abscores[:,j]+=ee.abs().sum((0,2),dtype=torch.float64)
   perstepcount+=len(y)*y.shape[-1]
 result=dict(trained=dict(mse=se/count,mae=ae/count),base=dict(mse=bse/count,mae=bae/count))
 if calibrate:
  idx=scores.argmin(1);result['calibration']=dict(weights=grid[idx].cpu().tolist(),grid=grid.cpu().tolist(),validation_mse=float(scores.gather(1,idx[:,None]).mean()/perstepcount),validation_mae=float(abscores.gather(1,idx[:,None]).mean()/perstepcount),selection_split='validation',test_labels_used_for_selection=False)
 return result

@torch.no_grad()
def evaluate_test(model,data,bank,h,batch,device,weight):
 se=ae=bse=bae=rse=rae=0.;count=windows=0;model.eval();start=time.monotonic()
 for x,y,o,_ in loader(data,h,'test',batch):
  x=x.to(device);y=y.to(device);o=o.to(device);base,diff,f=model.decompose(x,o,bank);raw,delta=model.mix(base,diff,f);pred=base+delta*weight[None,:,None];e=pred-y;re=raw-y;be=base-y
  se+=float(e.square().sum(dtype=torch.float64));ae+=float(e.abs().sum(dtype=torch.float64));rse+=float(re.square().sum(dtype=torch.float64));rae+=float(re.abs().sum(dtype=torch.float64));bse+=float(be.square().sum(dtype=torch.float64));bae+=float(be.abs().sum(dtype=torch.float64));count+=y.numel();windows+=len(y)
 assert windows==len(data.origins(h,'test'))
 return dict(trained=dict(mse=se/count,mae=ae/count),uncalibrated=dict(mse=rse/count,mae=rae/count),base=dict(mse=bse/count,mae=bae/count),windows=windows,inference_seconds=time.monotonic()-start)

from diagnostics import evaluate_test

def run(job,device):
 started=time.monotonic();torch.cuda.reset_peak_memory_stats();data,model,bank,graph,ck,bank_o=setup(job,device);base_hash=sha(ck);h=job['horizon'];ds=job['dataset'];v=job['variant'];batch=16 if ds=='PEMS08' else 4
 p=result_path(job);p.parent.mkdir(parents=True,exist_ok=True);cache=ROOT/'work_cache'/job['id'];train_o=data.origins(h,'train');val_o=data.origins(h,'val')
 if shutil.disk_usage(ROOT).free<45*2**30:raise RuntimeError('disk safety reserve: need at least45GiB free')
 for split,origins in [('train',train_o),('val',val_o)]:create_cache(cache/split,data,origins,h,model,bank,batch,device,job['id'])
 train=cached_loader(cache/'train',data,train_o,h,batch,True);val=cached_loader(cache/'val',data,val_o,h,batch,False)
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.002,weight_decay=1e-4);epochs=20;scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs) if ds=='PEMS08' else None
 best=float('inf');bad=0;history=[];checkpoint=p.parent/'adapter.pt'
 for epoch in range(1,epochs+1):
  st=time.monotonic();model.train();loss_sum=0.;steps=0
  for b,d,f,y in train:
   b=b.to(device);d=d.to(device);f=f.to(device);y=y.to(device);opt.zero_grad(set_to_none=True);pred,_=model.mix(b,d,f);loss=(pred-y).square().mean()+1e-4*torch.tanh(model.residual_scale).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('nonfinite ablation objective')
   loss.backward();torch.nn.utils.clip_grad_norm_(params,5);opt.step();loss_sum+=float(loss.detach());steps+=1
  if scheduler:scheduler.step()
  vm=evaluate_cached(model,val,device);history.append(dict(epoch=epoch,train=loss_sum/steps,validation=vm));log('ABLATION_EPOCH',job=job['id'],epoch=epoch,train_mse=loss_sum/steps,validation=vm,seconds=time.monotonic()-st)
  if vm['trained']['mse']<best:
   best=vm['trained']['mse'];bad=0;state={k:value.detach().cpu().clone() for k,value in model.state_dict().items() if not k.startswith('base.')};torch.save(dict(adapter=state,epoch=epoch,variant=v),checkpoint)
  else:bad+=1
  if bad>=5:break
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True);current=model.state_dict();current.update(saved['adapter']);model.load_state_dict(current,strict=True)
 raw_val=evaluate_cached(model,val,device,calibrate=True);cal=raw_val.pop('calibration')
 # Align every horizon with the main-table validation-only per-step grid.
 selected=cal['weights']
 weight=torch.tensor(selected,device=device);vm=evaluate_cached(model,val,device,weight)
 release=dict(adapter_sha256=sha(checkpoint),base_sha256=base_hash,variant=v,weights=selected,validation=vm,raw_validation=raw_val,test_labels_used_for_selection=False)
 write_json(p.parent/'test_release.json',release)
 tm=None if job.get('validation_only') else evaluate_test(model,data,bank,h,batch,device,weight)
 assert sha(ck)==base_hash and sha(checkpoint)==release['adapter_sha256']
 out=dict(job=job,status='COMPLETE',dataset=ds,horizon=h,seed=job['seed'],variant=v,split='validation' if job.get('validation_only') else 'test',protocol=data.metadata,graph=graph['metadata'],validation=vm,raw_validation=raw_val,calibration=cal,applied_weights=selected,test=tm,best_epoch=saved['epoch'],history=history,
          trainable_parameters=sum(x.numel() for x in params),base_checkpoint=str(ck),base_checkpoint_sha256=base_hash,memory_episodes=0 if v in ('periodic_only','residual_mlp') else len(bank_o),candidate_memory_budget=len(bank_o),memory_max_target_end=int(bank_o[-1]+h),memory_train_only=True,test_labels_used_for_selection=False,
          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,seconds=time.monotonic()-started,cache_math='frozen base/retrieval features computed once; same trainable forward and gradients as on-the-fly adapter')
 out['memory_storage']=memory_storage(bank);out['recipe_anchor']='main-table validation calibration; original fit budget and frozen paired backbone'
 write_json(p,out);log('ABLATION_RESULT',job=job['id'],test=tm,seconds=out['seconds'])
 # Only delete disposable files created by this completed job inside this root.
 del train,val
 if cache.resolve().is_relative_to((ROOT/'work_cache').resolve()):shutil.rmtree(cache)
