"""Online, resumable adapter fitting with the frozen STID recipe."""
import argparse,json,os,time,random,traceback
from pathlib import Path
import numpy as np
import torch
from common import ROOT,arrays,sha,write_json,check_protected,require_gpu,seed_all
from experiment import loader,evaluate,calibrate,save_checkpoint
from evaluate_locked import restore_strong,restore_adapter_state,assert_metrics_match
from train_strong_kv import RECIPE,capture_base,check_base
from physics_kv import StrongBackbonePhysicsKV,periodic_prior,trainable_parameters
from efficient_adapter import EfficientStrongKV
from support_graphs import tensors
from memory_budget import build_bank
from diagnostics_support import diagnostics,latency

def setup(job,device):
 a=arrays();protocol=json.loads((ROOT/'data/processed/protocol.json').read_text())
 base,audit=restore_strong(ROOT,a,protocol,'stid',2024,device)
 gd=json.loads((ROOT/f'graphs/{job["graph"]}.json').read_text());g=tensors(gd['edges'],716,gd['metadata'])
 cutoff=int(a['origins_val'][0])-13;daily,weekly=periodic_prior(a['data'][:cutoff],96,672)
 model=EfficientStrongKV(base,g,daily.to(device),weekly.to(device),12,96).to(device)
 bank,sampling=build_bank(a,g,model.daily,model.weekly,device)
 if job['kind']=='reuse':
  expected=json.loads((ROOT/'references/physical_validation_seed2024.json').read_text())
  assert g['metadata']['original_physical_metadata']['edges_sha256']==expected['physical_hypergraph']['edges_sha256']
  assert sampling['selected_origins_sha256']==expected['sampling']['selected_origins_sha256']
  payload=torch.load(ROOT/'references/physical_adapter_seed2024.pt',map_location='cpu',weights_only=False)
  restore_adapter_state(model,payload['adapter']);model.to(device)
  torch.testing.assert_close(model.incidence,g['incidence'].to(device),rtol=0,atol=0)
  weights=expected['horizon_gate']['weights']
 else:weights=None
 return model,bank,a,protocol,g,sampling,audit,weights

def saved_rng():
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(d):
 random.setstate(d['python']);np.random.set_state(d['numpy']);torch.set_rng_state(d['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in d['cuda']])

def preflight(job,device,gpu):
 seed_all();model,bank,a,protocol,g,sampling,audit,weights=setup(job,device)
 frozen=capture_base(model.base);x,y,t=next(iter(loader(a,a['origins_val'])))
 x=x.to(device);y=y.to(device);t=t.to(device)
 if job['kind']=='fit':
  model.eval()
  with torch.no_grad():
   pred,info=model(x,t,bank);torch.testing.assert_close(pred,info['base'],rtol=0,atol=0)
  model.train();opt=torch.optim.AdamW(trainable_parameters(model),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
  pred,_=model(x[:32],t[:32],bank)
  from metrics import masked_mae_loss
  loss=masked_mae_loss(pred,y[:32],float(a['mean']),float(a['std']))
  loss.backward()
  assert torch.isfinite(loss) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in trainable_parameters(model))
  opt.step();check_base(model.base,frozen)
  with torch.no_grad():
   pred,info=model(x,t,bank,disable_memory=True);torch.testing.assert_close(pred,info['base'],rtol=0,atol=0)
 else:
  with torch.no_grad():
   fast,fi=model(x[:2],t[:2],bank,residual_weight=weights)
   original,oi=StrongBackbonePhysicsKV.forward(model,x[:2],t[:2],bank,residual_weight=weights)
   torch.testing.assert_close(fast,original,rtol=1e-6,atol=1e-6)
   torch.testing.assert_close(fi['physical'],oi['physical'],rtol=1e-5,atol=1e-6)
  scores=evaluate(model,a,a['origins_val'],device,bank,weights)
  expected=json.loads((ROOT/'references/physical_validation_seed2024.json').read_text())
  assert_metrics_match(scores['kv'],expected['metrics']['kv'],'exact original physical validation')
  assert_metrics_match(scores['base'],expected['metrics']['base'],'exact paired physical baseline')
  check_base(model.base,frozen)
 report=dict(job=job,status='PASS',physical_gpu=gpu,sampling=sampling,base_restore=audit['validation_restore'],
  zero_start_checked=job['kind']=='fit',finite_backward_checked=job['kind']=='fit',base_unchanged=True,
  original_physical_replay_checked=job['kind']=='reuse',test_evaluated=False)
 write_json(ROOT/f'audit/preflight_{job["id"]}.json',report)
 print('SUPPORT_PREFLIGHT '+json.dumps(report),flush=True)

def run(job,device,gpu):
 folder=ROOT/'results'/job['id'];folder.mkdir(parents=True,exist_ok=True)
 result_path=folder/'result.json'
 if result_path.exists():
  d=json.loads(result_path.read_text());assert d['status']=='COMPLETE' and not d['test_evaluated'];return
 protected=check_protected();seed_all();model,bank,a,protocol,g,sampling,base_audit,weights=setup(job,device)
 frozen=capture_base(model.base);checkpoint=ROOT/'runs/stid/seed2024/model.pt';base_hash=sha(checkpoint)
 baseline_ref=json.loads((ROOT/'results/stid_validation_seed2024.json').read_text())
 baseline=evaluate(model,a,a['origins_val'],device,bank,weights)
 assert_metrics_match(baseline['base'],baseline_ref['metrics'],'paired frozen STID')
 rows=[];best_epoch=0;started=time.monotonic();resumed=False
 if job['kind']=='fit':
  assert_metrics_match(baseline['kv'],baseline_ref['metrics'],'zero-start STID')
  opt=torch.optim.AdamW(trainable_parameters(model),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
  best=float('inf');bad=0;first=1;best_state=None;last=folder/'resume.pt'
  if last.exists():
   state=torch.load(last,map_location=device,weights_only=False)
   assert state['base_hash']==base_hash and state['graph_hash']==g['metadata']['edges_sha256'] and state['recipe']==RECIPE
   restore_adapter_state(model,state['adapter']);opt.load_state_dict(state['optimizer']);best=state['best'];bad=state['bad'];best_epoch=state['best_epoch'];best_state=state['best_state'];rows=state['rows'];first=state['epoch']+1;restore_rng(state['rng']);resumed=True
  from metrics import masked_mae_loss
  for epoch in range(first,RECIPE['max_epochs']+1):
   if bad>=RECIPE['patience']:break
   model.train();tick=progress=time.monotonic();total=steps=0
   for x,y,t in loader(a,a['origins_train'],32,True):
    x=x.to(device);y=y.to(device);t=t.to(device);opt.zero_grad(set_to_none=True)
    pred,_=model(x,t,bank);loss=masked_mae_loss(pred,y,float(a['mean']),float(a['std']))
    loss=loss+RECIPE['scale_regularization']*torch.tanh(model.residual_scale).square().mean()
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(trainable_parameters(model),RECIPE['clip_grad']);opt.step();total+=float(loss.detach());steps+=1
    if time.monotonic()-progress>30:
     print(json.dumps(dict(stage='TRAIN_PROGRESS',job=job['id'],epoch=epoch,batch=steps)),flush=True);progress=time.monotonic()
   scores=evaluate(model,a,a['origins_val'],device,bank);value=scores['kv']['raw_masked']['mae'];check_base(model.base,frozen)
   row=dict(epoch=epoch,train_mae=total/steps,validation=scores['kv']['horizon_average'],selection_mae=value,seconds=time.monotonic()-tick);rows.append(row)
   if value<best:
    best=value;best_epoch=epoch;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('base.')}
   else:bad+=1
   adapter={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('base.')}
   save_checkpoint(last,dict(epoch=epoch,adapter=adapter,optimizer=opt.state_dict(),best=best,bad=bad,best_epoch=best_epoch,best_state=best_state,rows=rows,rng=saved_rng(),base_hash=base_hash,graph_hash=g['metadata']['edges_sha256'],recipe=RECIPE))
   write_json(ROOT/f'status/{job["id"]}.json',dict(state='RUNNING',job=job,epoch=epoch,best_epoch=best_epoch,validation_mae=value,physical_gpu=gpu,updated=time.time()))
   print('SUPPORT_EPOCH '+json.dumps(dict(job=job['id'],**row)),flush=True)
  if best_state is None:raise RuntimeError('No validation-selected adapter')
  restore_adapter_state(model,best_state)
  uncalibrated=evaluate(model,a,a['origins_val'],device,bank)
  calibration=calibrate(model,a,a['origins_val'],bank,device);weights=calibration['weights']
  save_checkpoint(folder/'adapter.pt',dict(adapter=best_state,recipe=RECIPE,protocol=protocol,seed=2024,best_epoch=best_epoch,
    graph_sha256=g['metadata']['edges_sha256'],base_checkpoint_sha256=base_hash,sampling=sampling,horizon_gate=calibration,test_evaluated=False))
 else:
  reference=json.loads((ROOT/'references/physical_validation_seed2024.json').read_text());best_epoch=reference['best_epoch'];calibration=reference['horizon_gate'];uncalibrated=reference['uncalibrated']
 scores=evaluate(model,a,a['origins_val'],device,bank,weights)
 assert_metrics_match(scores['base'],baseline_ref['metrics'],'final paired STID')
 if job['kind']=='reuse':assert_metrics_match(scores['kv'],reference['metrics']['kv'],'reused physical model')
 mechanistic=diagnostics(model,bank,a,weights,g)
 mechanistic['aggregate_masked_mse_gain']=scores['base']['raw_masked']['mse']-scores['kv']['raw_masked']['mse']
 timing=latency(model,bank,a,weights)
 timing['incidence_and_pool_gpu_bytes']=sum(v.numel()*v.element_size() for v in [model.incidence,model.edge_pool])
 check_base(model.base,frozen);assert sha(checkpoint)==base_hash
 assert check_protected()==protected
 result=dict(status='COMPLETE',dataset='LargeST-SD',backbone='STID',job=job,split='validation',seed=2024,seq_len=12,horizon=12,
  physical_gpu=gpu,protocol=protocol,recipe=RECIPE,metrics=scores,uncalibrated=uncalibrated,horizon_gate=calibration,best_epoch=best_epoch,epochs=rows,
  graph=g['metadata'],sampling=sampling,diagnostics=mechanistic,latency=timing,base_checkpoint_sha256=base_hash,
  base_parameters_and_buffers_unchanged=True,memory_train_only=True,test_evaluated=False,test_labels_used_for_selection=False,
  reference_reused=job['kind']=='reuse',resumed=resumed,seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
  interpretation='Exploratory single-training-seed validation study; structure seeds are separate repetitions, not training seeds.')
 write_json(result_path,result);write_json(ROOT/f'status/{job["id"]}.json',dict(state='COMPLETE',job=job,updated=time.time(),physical_gpu=gpu))
 print('SUPPORT_RESULT '+json.dumps(dict(job=job['id'],metrics=scores['kv']['horizon_average'],best_epoch=best_epoch)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
 jobs=json.loads((ROOT/'audit/jobs.json').read_text());job=next(j for j in jobs if j['id']==args.job)
 gpu,device=require_gpu();check_protected()
 if args.preflight:preflight(job,device,gpu)
 else:run(job,device,gpu)

if __name__=='__main__':
 try:main()
 except Exception as e:
  print('SUPPORT_FAILURE '+repr(e),flush=True);traceback.print_exc();raise
