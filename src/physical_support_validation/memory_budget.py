"""Byte-matched train-only memory, with the original phase-balanced sampler."""
import hashlib,math
import numpy as np
from common import LENGTH,HORIZON,PERIOD

def storage_bytes(episodes,edges):
 # values, raw keys, normalized keys, key mean/std, and int64 timestamps.
 return episodes*edges*4*(HORIZON+2*13)+edges*2*13*4+episodes*8

def episode_budget(edges,reference_edges=685,reference_episodes=1749):
 target=storage_bytes(reference_episodes,reference_edges)
 count=(target-edges*2*13*4)//(edges*4*(HORIZON+2*13)+8)
 if count<8:raise RuntimeError('Support count exceeds meaningful matched-memory budget')
 assert storage_bytes(count,edges)<=target
 return int(count),target

def select_count(origins,budget,period=96):
 from memory_sampling import _phase_order
 x=np.asarray(origins,dtype=np.int64)
 assert len(x)>0 and np.all(np.diff(x)>0) and 0<budget<=len(x)
 if budget==len(x):return x.copy()
 phase=x%period;capacities=np.bincount(phase,minlength=period);order=_phase_order(period);order=order[capacities[order]>0]
 quotas=np.zeros(period,dtype=np.int64);remaining=budget
 while remaining:
  take=order[quotas[order]<capacities[order]][:remaining]
  if not len(take):raise RuntimeError('Insufficient training origins')
  quotas[take]+=1;remaining-=len(take)
 picked=[];gold=(math.sqrt(5)-1)/2
 for rank,p in enumerate(order):
  q=int(quotas[p])
  if q:
   eligible=x[phase==p];offset=((rank+.5)*gold)%1
   slots=np.floor((np.arange(q)+offset)*len(eligible)/q).astype(np.int64);picked.append(eligible[slots])
 result=np.sort(np.concatenate(picked));assert len(result)==budget and len(np.unique(result))==budget
 return result

def actual_storage(bank):
 import torch
 unique={};tensors={}
 for k,v in vars(bank).items():
  if torch.is_tensor(v):
   s=v.untyped_storage();unique[(str(v.device),s.data_ptr())]=s.nbytes();tensors[k]=dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size())
 return dict(unique_tensor_bytes=sum(unique.values()),tensors=tensors)

def build_bank(arrays,graph,daily,weekly,device):
 import torch
 from physics_kv import residual_history,residual_future,edge_descriptors
 from memory_sampling import build_phase_balanced_bank,select_memory_origins
 cutoff=int(arrays['origins_val'][0])-LENGTH-1;assert cutoff==21009
 train=arrays['origins_train'];eligible=train[train+HORIZON<=cutoff]
 budget,target=episode_budget(len(graph['edges']))
 if budget>len(eligible):raise RuntimeError('Cannot match KV bytes without duplicating episodes')
 chosen=select_count(eligible,budget,PERIOD)
 if len(graph['edges'])==685:
  expected=select_memory_origins(eligible,12,96,horizon=12,train_end=cutoff,seq_len=12)
  assert np.array_equal(chosen,expected),'Physical budget sampler differs from original'
 keys=[];values=[];pool=graph['edge_pool'].to(device)
 for i in range(0,len(chosen),32):
  oo=chosen[i:i+32];t=torch.as_tensor(oo,device=device)
  x=torch.from_numpy(np.stack([arrays['data'][s-12:s] for s in oo])).to(device)
  y=torch.from_numpy(np.stack([arrays['data'][s:s+12] for s in oo])).to(device)
  with torch.no_grad():
   rh=residual_history(x,t,daily,weekly);rv=residual_future(y,t,daily,weekly)
   keys.append(edge_descriptors(rh,t,pool,period=96).cpu());values.append(torch.einsum('bhn,ne->beh',rv,pool).cpu())
 bank=build_phase_balanced_bank(torch.as_tensor(chosen),torch.cat(keys),torch.cat(values),device,8,.5)
 measured=actual_storage(bank);expected=storage_bytes(budget,len(graph['edges']))
 assert measured['unique_tensor_bytes']==expected,(measured['unique_tensor_bytes'],expected)
 assert expected<=target
 assert (target-expected)/target<.005,'Integer episode budget differs by more than 0.5 percent'
 meta=dict(memory_episodes=budget,supports=len(graph['edges']),target_kv_bytes=target,actual_kv_bytes=expected,
  budget_relative_difference=(expected-target)/target,selection='Original timestamp-only phase-balanced stratification with exact byte-controlled count',
  selected_origins_sha256=hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest(),
  first_origin=int(chosen[0]),last_origin=int(chosen[-1]),train_end=cutoff,last_target_exclusive=int(chosen[-1]+12),
  phase_counts=np.bincount(chosen%96,minlength=96).tolist(),memory_train_only=True,
  eligible_training_origins=len(eligible),density_note='Binary/singleton episode density changes intentionally to match actual KV tensor bytes; not a same-density comparison',
  chronology='episode_origin + 12 <= query_origin - 12',storage=measured)
 assert meta['last_target_exclusive']<=cutoff
 return bank,meta
