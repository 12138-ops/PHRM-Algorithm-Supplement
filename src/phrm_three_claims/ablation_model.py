"""Retrainable controls for the frozen PhysicsResidualKV mechanism."""
import csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import frozen_physics as frozen
from frozen_coverage_graph import load_graph as coverage_graph
from common import PROJECTS,digest

VARIANTS=('full','periodic_only','node_level_memory','random_support','physical_support','support_shuffled','residual_mlp','global_memory','pairwise_graph','random_hypergraph','raw_memory','clock_only','fixed_gate')

def tensors(edges,n,metadata=None):
 incidence=torch.zeros(n,len(edges),dtype=torch.float32)
 for i,e in enumerate(edges):incidence[list(e),i]=1
 if not (incidence.sum(1)>0).all():raise ValueError('graph leaves uncovered nodes')
 return dict(incidence=incidence,pool=incidence/incidence.sum(1,keepdim=True),edge_pool=incidence/incidence.sum(0,keepdim=True),edges=[list(e) for e in edges],num_nodes=n,metadata=metadata or {})

def rewired_graph(graph,seed=19073):
 """Double-edge swaps preserve node degrees and hyperedge cardinalities exactly."""
 rng=random.Random(seed);edges=[set(e) for e in graph['edges']];old=[tuple(sorted(e)) for e in edges];seen=set(old)
 swaps=0;attempts=max(1000,40*sum(map(len,edges)))
 for _ in range(attempts):
  i,j=rng.sample(range(len(edges)),2);u=rng.choice(sorted(edges[i]));v=rng.choice(sorted(edges[j]))
  if u==v or v in edges[i] or u in edges[j]:continue
  aa=(edges[i]-{u})|{v};bb=(edges[j]-{v})|{u};ta,tb=tuple(sorted(aa)),tuple(sorted(bb))
  if ta==tb or ta in seen or tb in seen:continue
  seen.remove(tuple(sorted(edges[i])));seen.remove(tuple(sorted(edges[j])));seen.add(ta);seen.add(tb);edges[i]=aa;edges[j]=bb;swaps+=1
 result=tensors([sorted(e) for e in edges],graph['incidence'].shape[0],dict(control='degree-and-size-preserving incidence rewiring',seed=seed,swaps=swaps,changed_edges=sum(tuple(sorted(e))!=o for e,o in zip(edges,old))))
 assert torch.equal(result['incidence'].sum(1),graph['incidence'].sum(1))
 assert torch.equal(result['incidence'].sum(0),graph['incidence'].sum(0))
 if swaps<100:raise RuntimeError('random topology did not sufficiently mix')
 return result

def shuffled_graph(graph,seed=271828):
    """Global node-label permutation: preserves support sizes/degrees but breaks road membership."""
    rng=random.Random(seed); n=graph['incidence'].shape[0]; perm=list(range(n)); rng.shuffle(perm)
    edges=[[perm[u] for u in e] for e in graph['edges']]
    return tensors(edges,n,dict(control='global node permutation preserving support sizes and node degrees',seed=seed))

def graph_for(dataset,variant):
 if dataset=='PEMS08':
  p=PROJECTS/'tqnet_physics_residual_kv_pems08/dataset/physical/PEMS08_physical_candidates.json';g=frozen.load_graph(p,170);csv_path=PROJECTS/'TQNet_physical_attention_PEMS08/dataset/physical/PEMS08_edges.csv'
 elif dataset=='PEMS07':
  csv_path=PROJECTS/'PhysicsResidualKV_MultiDataset/dataset/physical/PEMS07.csv';g=coverage_graph(csv_path,883)
 else:raise ValueError(dataset)
 if variant in ('random_hypergraph','random_support'):g=rewired_graph(g)
 if variant=='support_shuffled':g=shuffled_graph(g)
 if variant=='physical_support': pass
 if variant=='node_level_memory':
  # Match the physical support count when possible while retaining node-level retrieval.
  target=len(g['edges']); n=g['incidence'].shape[0]
  if target >= n: edges=[[i] for i in range(n)] + [[i % n] for i in range(target-n)]
  else: edges=[[i] for i in range(n)]
  g=tensors(edges,n,dict(control='singleton node supports',storage_match_target_edges=target,storage_match_exact=(len(edges)==target)))
 if variant=='pairwise_graph':
  with csv_path.open() as f:rows=list(csv.reader(f))
  pairs=sorted({tuple(sorted(map(int,row[:2]))) for row in rows[1:] if row and row[0]!=row[1]})
  g=tensors(pairs,g['incidence'].shape[0],dict(control='all genuine binary road edges',source=str(csv_path),budget_note='edge count differs from hypergraph to preserve all genuine roads and node coverage'))
 m=dict(g.get('metadata',{}));m.update(num_nodes=g['incidence'].shape[0],edges=len(g['edges']),higher_order_edges=sum(len(e)>2 for e in g['edges']),covered_nodes=int((g['incidence'].sum(1)>0).sum()),edge_sizes=[len(e) for e in g['edges']],incidence_sha256=hashlib.sha256(g['incidence'].numpy().tobytes()).hexdigest())
 m.update(physical_graph_used=variant in ('full','physical_support'),effective_memory_groups=0 if variant in ('periodic_only','residual_mlp') else 1 if variant=='global_memory' else len(g['edges']),value_channels='node trajectories' if variant=='node_level_memory' else 'none' if variant in ('periodic_only','residual_mlp') else 'hyperedge trajectories')
 g['metadata']=m;return g

class ClockBank(frozen.PhysicalKVBank):
 def __init__(self,origins,keys,values,device,topk,temperature):
  keys=keys.clone();keys[...,:11]=0
  super().__init__(origins,keys,values,device,topk,temperature)
 def retrieve(self,query,origins,seq_len):
  query=query.clone();query[...,:11]=0
  return super().retrieve(query,origins,seq_len)

class GlobalBank:
 def __init__(self,origins,keys,values,device,topk,temperature):
  self.origins=origins.to(device);self.values=values.to(device);self.mean=keys.mean(0).to(device);self.std=keys.std(0,unbiased=False).clamp_min(1e-3).to(device);self.keys=(keys.to(device)-self.mean)/self.std;self.topk=topk;self.temperature=temperature
 def retrieve_global(self,query,origins,seq_len):
  q=(query-self.mean)/self.std
  dist=(q.square().sum(-1)[:,None]+self.keys.square().sum(-1)[None,:]-2*q@self.keys.T).clamp_min(0)
  allowed=self.origins[None,:]+self.values.shape[1]<=origins[:,None]-seq_len
  dist=dist.masked_fill(~allowed,float('inf'));nearest,ids=dist.topk(min(self.topk,len(self.origins)),largest=False)
  valid=torch.isfinite(nearest);weights=torch.softmax(-nearest.masked_fill(~valid,1e6)/self.temperature,-1)*valid;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
  return (weights[:,:,None,None]*self.values[ids]).sum(1)

@torch.no_grad()
def build_bank(data,origins,h,graph,daily,weekly,device,batch,variant,topk=8):
 keys=[];values=[];pool=graph['edge_pool'].to(device)
 for start in range(0,len(origins),batch):
  oo=origins[start:start+batch];o=torch.tensor(oo,device=device)
  x=torch.from_numpy(np.stack([data.values[t-96:t] for t in oo])).to(device);y=torch.from_numpy(np.stack([data.values[t:t+h] for t in oo])).to(device)
  rh=frozen.residual_history(x,o,daily,weekly);rv=frozen.residual_future(y,o,daily,weekly)
  if variant=='global_memory':
   global_pool=pool.new_full((pool.shape[0],1),1/pool.shape[0]);keys.append(frozen.edge_descriptors(rh,o,global_pool,period=daily.shape[0])[:,0].cpu());values.append(rv.cpu())
  else:
   keys.append(frozen.edge_descriptors(rh,o,pool,period=daily.shape[0]).cpu());values.append(torch.einsum('bhn,ne->beh',rv,pool).cpu())
 cls=GlobalBank if variant=='global_memory' else ClockBank if variant=='clock_only' else frozen.PhysicalKVBank
 return cls(torch.as_tensor(origins),torch.cat(keys),torch.cat(values),device,topk,.5)

class Adapter(frozen.TQNetPhysicsKV):
 def __init__(self,base,graph,daily,weekly,h,variant):
  super().__init__(base,graph,daily,weekly,h,96);self.variant=variant
  if variant=='fixed_gate':
   for p in self.gate.parameters():p.requires_grad_(False)
   self.fixed_logits=nn.Parameter(torch.full((h,),-2.))
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def decompose(self,x,origins,bank):
  if self.variant in ('periodic_only','residual_mlp','global_memory'):
   base=self.base(x,origins.remainder(self.base.cycle_len));periodic=frozen._phase_value(self.daily,self.weekly,origins,torch.arange(self.horizon,device=x.device))
   node_memory=torch.zeros_like(base)
   if self.variant=='global_memory':
    rh=frozen.residual_history(x,origins,self.daily,self.weekly);pool=x.new_full((x.shape[-1],1),1/x.shape[-1]);query=frozen.edge_descriptors(rh,origins,pool,period=self.daily.shape[0])[:,0];node_memory=bank.retrieve_global(query,origins,x.shape[1])
   physical=periodic+node_memory;diff=physical-base
  else:
   _,info=frozen.TQNetPhysicsKV.forward(self,x,origins,bank)
   base=info['base'];diff=info['diff'];periodic=frozen._phase_value(self.daily,self.weekly,origins,torch.arange(self.horizon,device=x.device));node_memory=info['physical']-periodic
  features=torch.stack([base.mean(1),base.std(1,unbiased=False),diff.mean(1),diff.std(1,unbiased=False),x[:,-1],node_memory.mean(1)],-1)
  if self.variant=='residual_mlp':
   features=torch.stack([base.mean(1),base.std(1,unbiased=False),x[:,-1],x.mean(1),x.std(1,unbiased=False),x[:,-1]-x[:,-12:].mean(1)],-1);diff=torch.ones_like(base)
  return base,diff,features
 def mix(self,base,diff,features,weight=None):
  if self.variant=='fixed_gate':gate=self.fixed_logits.sigmoid()[None,:,None]
  elif self.variant=='residual_mlp':gate=self.gate(features).transpose(1,2)
  else:gate=self.gate(features).sigmoid().transpose(1,2)
  delta=torch.tanh(self.residual_scale)[None,:,None]*gate*diff
  if weight is not None:delta=delta*weight[None,:,None]
  return base+delta,delta
 def forward(self,x,origins,bank,weight=None):
  base,diff,features=self.decompose(x,origins,bank);return self.mix(base,diff,features,weight)
