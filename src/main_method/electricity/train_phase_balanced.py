"""Sampling/clock repair, isolated from all running Solar and PEMS jobs."""
from __future__ import annotations
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from generic_graph import load_graph
from memory_sampling import select_memory_origins,sampling_metadata,build_phase_balanced_bank
from physics_kv import TQNetPhysicsKV,periodic_prior,residual_history,residual_future,edge_descriptors,trainable_parameters
from protocol import load_data,SPECS
from settings import UPSTREAM
from training_common import Windows,load_base,evaluate,calibrate

PHASE_VERSION='phase-balanced-kv-v1'

def stage(name,**kw):print('PHASE_STAGE '+json.dumps(dict(stage=name,time=time.time(),**kw)),flush=True)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()

def validation_partition(prepared,horizon,policy):
 _,origins,_=prepared.origins(horizon)
 if policy=='full':return origins,origins,dict(policy=policy,selection_and_audit_same=True)
 if policy!='blocked':raise ValueError(policy)
 boundary=prepared.train_end+(prepared.val_end-prepared.train_end)//2
 selection=origins[origins+horizon<=boundary];audit=origins[origins>=boundary]
 if not len(selection) or not len(audit):raise ValueError('insufficient chronological validation span')
 assert selection[-1]+horizon<=audit[0]
 return selection,audit,dict(policy=policy,boundary=boundary,selection_windows=len(selection),audit_windows=len(audit),purged_origins=int(len(origins)-len(selection)-len(audit)),selection_target_end=int(selection[-1]+horizon),audit_target_start=int(audit[0]),selection_and_audit_same=False,base_checkpoint_selected_using_original_full_validation=True)

def loader(data,origins,horizon,batch_size,shuffle=False):
 return DataLoader(Windows(data,origins,horizon),batch_size=batch_size,shuffle=shuffle,num_workers=0,pin_memory=True)

@torch.no_grad()
def build_bank(data,origins,horizon,graph,daily,weekly,batch_size,device):
 keys=[];values=[];pool=graph['edge_pool'].to(device);last=time.monotonic()
 stage('memory_start',episodes=len(origins),horizon=horizon)
 for start in range(0,len(origins),batch_size):
  ts_np=origins[start:start+batch_size];ts=torch.as_tensor(ts_np,device=device,dtype=torch.long)
  x=torch.stack([torch.from_numpy(data[int(t)-96:int(t)]) for t in ts_np]).to(device)
  y=torch.stack([torch.from_numpy(data[int(t):int(t)+horizon]) for t in ts_np]).to(device)
  rh=residual_history(x,ts,daily,weekly);rv=residual_future(y,ts,daily,weekly)
  keys.append(edge_descriptors(rh,ts,pool,period=daily.shape[0]).cpu());values.append(torch.einsum('bhn,ne->beh',rv,pool).cpu())
  if time.monotonic()-last>=30:stage('memory_progress',completed=start+len(ts_np),total=len(origins));last=time.monotonic()
 bank=build_phase_balanced_bank(torch.as_tensor(origins),torch.cat(keys),torch.cat(values),device,8,.5)
 stage('memory_ready',episodes=len(origins),clock_mean=bank.mean[0,0,11:].tolist(),clock_std=bank.std[0,0,11:].tolist())
 return bank

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--dataset',required=True,choices=['ETTh1','ETTm1','ETTm2','Electricity']);p.add_argument('--data_path',required=True);p.add_argument('--edge_csv',required=True);p.add_argument('--base_checkpoint',required=True);p.add_argument('--num_nodes',type=int,required=True);p.add_argument('--horizon',type=int,required=True);p.add_argument('--cycle',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--split',required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=5);p.add_argument('--batch_size',type=int,default=4);p.add_argument('--memory_stride',type=int,default=12);p.add_argument('--device',default='cuda:0');p.add_argument('--runs_root',default='runs_phase_balanced');p.add_argument('--validation_only',action='store_true');p.add_argument('--validation_policy',choices=['full','blocked'],default='full');p.add_argument('--phase_version',choices=[PHASE_VERSION],default=PHASE_VERSION);a=p.parse_args()
 if a.dataset.startswith('ETT') and (not a.validation_only or a.seed!=2024 or a.horizon not in [96,720]):raise RuntimeError('ETT repair screen is six seed2024 validation-only cells')
 if a.dataset.startswith('ETT') and a.validation_policy!='blocked':raise RuntimeError('ETT repair screen requires predefined blocked validation')
 if a.dataset=='Electricity' and (a.validation_only or a.validation_policy!='full'):raise RuntimeError('Electricity must retain original full validation/test protocol')
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 root=Path(a.root).resolve();assert root==Path(__file__).resolve().parent
 if Path(a.runs_root).is_absolute() or '..' in Path(a.runs_root).parts:raise ValueError('outputs must remain within repair project')
 output=root/a.runs_root/a.dataset/f'h{a.horizon}'/f'seed{a.seed}'
 if (output/'result.json').exists():raise FileExistsError('completed repair cell already exists')
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 spec=SPECS[a.dataset];assert (a.num_nodes,a.cycle,a.split)==(spec['nodes'],spec['cycle'],spec['split'])
 prepared=load_data(a.dataset,a.data_path);expected=json.loads((root/'audit/preflight.json').read_text());assert expected['status']=='PASS' and expected['repair_version']==PHASE_VERSION
 assert expected['datasets'][a.dataset]['protocol']['fingerprint']==prepared.metadata['fingerprint']
 data=prepared.data[:prepared.val_end] if a.validation_only else prepared.data
 device=torch.device(a.device);started=time.monotonic();base_hash=sha(a.base_checkpoint)
 daily,weekly=periodic_prior(data[:prepared.train_end],spec['daily'],spec['weekly']);daily=daily.to(device);weekly=weekly.to(device)
 graph=load_graph(Path(a.edge_csv),a.num_nodes);base=load_base(a.base_checkpoint,a.num_nodes,a.horizon,a.cycle,device,use_revin=spec['use_revin'],dropout=spec.get('dropout',0));model=TQNetPhysicsKV(base,graph,daily,weekly,a.horizon,96).to(device)
 train_o,full_val_o,test_o=prepared.origins(a.horizon);selection_o,audit_o,partition=validation_partition(prepared,a.horizon,a.validation_policy)
 bank_o=select_memory_origins(train_o,a.memory_stride,spec['daily'],horizon=a.horizon,train_end=prepared.train_end)
 sampling=sampling_metadata(train_o,bank_o,spec['daily'],a.memory_stride,horizon=a.horizon,train_end=prepared.train_end)
 stage('protocol',dataset=a.dataset,horizon=a.horizon,seed=a.seed,visible_gpu=torch.cuda.current_device(),validation_only=a.validation_only,phase_version=PHASE_VERSION,protocol_fingerprint=prepared.metadata['fingerprint'],sampling=sampling,validation_partition=partition)
 bank=build_bank(data,bank_o,a.horizon,graph,daily,weekly,a.batch_size,device)
 train=loader(data,train_o,a.horizon,a.batch_size,True);selection=loader(data,selection_o,a.horizon,a.batch_size);audit=loader(data,audit_o,a.horizon,a.batch_size)
 opt=torch.optim.AdamW(trainable_parameters(model),lr=2e-3,weight_decay=1e-4);best=float('inf');best_state=None;best_epoch=bad=0
 for epoch in range(1,a.epochs+1):
  epoch_start=time.monotonic();last=epoch_start;model.train();model.base.eval();total=0.;steps=0
  for x,y,o in train:
   x=x.to(device).float();y=y.to(device).float();o=o.to(device).long();opt.zero_grad(set_to_none=True);pred,_=model(x,o,bank);loss=(pred-y).square().mean()+1e-4*torch.tanh(model.residual_scale).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(trainable_parameters(model),5);opt.step();total+=float(loss.detach());steps+=1
   if time.monotonic()-last>=30:stage('train_progress',epoch=epoch,batch=steps,total=len(train));last=time.monotonic()
  vm=evaluate(model,selection,bank,device)
  if not np.isfinite(vm['trained']['mse']):raise RuntimeError('non-finite validation loss')
  print(f"epoch={epoch} train={total/max(steps,1):.6f} selection_mse={vm['trained']['mse']:.6f} selection_mae={vm['trained']['mae']:.6f} base_mse={vm['base']['mse']:.6f} seconds={time.monotonic()-epoch_start:.1f}",flush=True)
  if vm['trained']['mse']<best:best=vm['trained']['mse'];best_epoch=epoch;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('base.')}
  else:bad+=1
  if bad>=a.patience:print('early_stop',flush=True);break
 if best_state is None:raise RuntimeError('no selected adapter')
 current=model.state_dict();current.update(best_state);model.load_state_dict(current,strict=True)
 calibration=calibrate(model,selection,bank,device);weights=torch.tensor(calibration['weights'],device=device)
 selection_metrics=evaluate(model,selection,bank,device,weights);audit_metrics=evaluate(model,audit,bank,device,weights)
 result=dict(model='TQNetPhysicsResidualKV',phase_version=PHASE_VERSION,dataset=a.dataset,horizon=a.horizon,seed=a.seed,split='validation' if a.validation_only else 'test',protocol=a.split,protocol_fingerprint=prepared.metadata['fingerprint'],preprocessing=prepared.metadata,train_end=prepared.train_end,val_end=prepared.val_end,memory_train_only=True,memory_max_target_end=int(bank_o[-1]+a.horizon),memory_episodes=len(bank_o),sampling=sampling,normalization=bank.normalization_metadata,test_labels_used_for_selection=False,test_evaluated=False,best_epoch=best_epoch,horizon_gate=calibration,validation_partition=partition,selection_validation=selection_metrics,validation_audit=audit_metrics,base_checkpoint=Path(a.base_checkpoint).name,base_checkpoint_sha256=base_hash,paired_same_checkpoint=True,physical_edges=len(graph['edges']),topology=spec['topology'],execution=dict(visible_gpu=torch.cuda.current_device(),workers=0,torch_threads=2),seconds=time.monotonic()-started)
 if a.validation_only:
  # Report this only after choices are frozen; it is not used for model selection.
  result['validation_full_diagnostic']=evaluate(model,loader(data,full_val_o,a.horizon,a.batch_size),bank,device,weights)
  prior=UPSTREAM/f'runs_corrected/{a.dataset}/h{a.horizon}/seed{a.seed}/result.json'
  old=json.loads(prior.read_text());result['old_full_validation_reference']=old['validation'];result['old_reference_selected_on_full_validation']=True
 else:
  result['validation']=selection_metrics
  result['test']=evaluate(model,loader(data,test_o,a.horizon,a.batch_size),bank,device,weights);result['test_evaluated']=True;result['windows']=len(test_o)
 assert sha(a.base_checkpoint)==base_hash
 output.mkdir(parents=True,exist_ok=True)
 torch.save(dict(adapter=best_state,phase_version=PHASE_VERSION,base_checkpoint=a.base_checkpoint,base_checkpoint_sha256=base_hash,best_epoch=best_epoch,protocol=prepared.metadata,sampling=sampling,normalization=bank.normalization_metadata,horizon_gate=calibration,validation_partition=partition),output/'adapter.pt')
 pending=output/'result.json.tmp';pending.write_text(json.dumps(result,indent=2));pending.replace(output/'result.json')
 print(('PHASE_BALANCED_VALIDATION_RESULT ' if a.validation_only else 'PHASE_BALANCED_KV_RESULT ')+json.dumps(result),flush=True)

if __name__=='__main__':main()
