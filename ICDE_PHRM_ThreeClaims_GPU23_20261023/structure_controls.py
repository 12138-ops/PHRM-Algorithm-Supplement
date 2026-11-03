"""Frozen train-only structure controls; no validation or test fitting."""
import hashlib,json,math,random
from collections import Counter
import numpy as np
import torch
from common import ROOT,sha,write_json
from ablation_model import graph_for as physical_graph,tensors,rewired_graph

VARIANTS=('statistical_hypergraph','node_independent_memory')
TOPOLOGY_SEED=19073
OPTIMIZATION_ATTEMPTS=200000

def correlation(train_residual):
 x=np.asarray(train_residual,dtype=np.float64)
 if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('finite training matrix required')
 x=x-x.mean(0,keepdims=True);norm=np.sqrt((x*x).sum(0));valid=norm>1e-12
 x=x/np.maximum(norm,1e-12);c=np.clip(x.T@x,-1.,1.)
 c[~valid,:]=0;c[:,~valid]=0;np.fill_diagonal(c,0.)
 return c

def coherence(edges,c):
 return float(np.mean([c[np.ix_(e,e)].sum()/(len(e)*(len(e)-1)) for e in edges]))

def optimize_statistics(physical,c,attempts=OPTIMIZATION_ATTEMPTS):
 """Degree-preserving swaps maximize mean within-support signed correlation.

 A heavily randomized incidence is the starting point. Physical memberships
 never enter the score; only fixed incidence margins define the capacity.
 """
 initial=rewired_graph(physical,TOPOLOGY_SEED)
 edges=[set(e) for e in initial['edges']];seen={tuple(sorted(e)) for e in edges}
 rng=random.Random(TOPOLOGY_SEED+1);before=coherence(initial['edges'],c);accepted=0
 for _ in range(attempts):
  i,j=rng.sample(range(len(edges)),2);u=rng.choice(sorted(edges[i]));v=rng.choice(sorted(edges[j]))
  if u==v or v in edges[i] or u in edges[j]:continue
  left=sorted(edges[i]-{u});right=sorted(edges[j]-{v})
  gain=(c[v,left].sum()-c[u,left].sum())/math.comb(len(edges[i]),2)+(c[u,right].sum()-c[v,right].sum())/math.comb(len(edges[j]),2)
  if gain<=1e-12:continue
  a=tuple(sorted(left+[v]));b=tuple(sorted(right+[u]))
  if a==b or a in seen or b in seen:continue
  seen.remove(tuple(sorted(edges[i])));seen.remove(tuple(sorted(edges[j])));seen.update((a,b));edges[i]=set(a);edges[j]=set(b);accepted+=1
 final=[sorted(e) for e in edges];after=coherence(final,c)
 assert after>=before-1e-10
 report=dict(randomization=initial['metadata'],optimization_seed=TOPOLOGY_SEED+1,optimization_attempts=attempts,accepted_improvements=accepted,initial_mean_signed_correlation=before,final_mean_signed_correlation=after,objective='equal-weight mean of within-support signed Pearson correlation on training periodic residuals',physical_adjacency_in_objective=False)
 return final,report

def graph_metadata(g):
 a=g['incidence'];return dict(num_nodes=a.shape[0],edges=a.shape[1],higher_order_edges=sum(len(e)>2 for e in g['edges']),covered_nodes=int((a.sum(1)>0).sum()),edge_sizes=[len(e) for e in g['edges']],node_degrees=a.sum(1).int().tolist(),incidence_sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest(),effective_memory_groups=a.shape[1],value_channels='hyperedge trajectories',physical_graph_used=False)

def prepare(dataset,data,daily,weekly):
 physical=physical_graph(dataset,'full');n=data.values.shape[1];end=data.metadata['train_end']
 # Do not read a single validation/test row while fitting the statistical graph.
 train=np.array(data.values[:end],copy=True);ids=np.arange(end)
 residual=train-daily.cpu().numpy()[ids%288]-weekly.cpu().numpy()[ids%2016]
 c=correlation(residual);edges,report=optimize_statistics(physical,c)
 candidates={tuple(e) for e in physical['edges']};stats=tensors(edges,n)
 assert torch.equal(stats['incidence'].sum(0),physical['incidence'].sum(0))
 assert torch.equal(stats['incidence'].sum(1),physical['incidence'].sum(1))
 assert len({tuple(e) for e in edges})==len(edges)
 report.update(physical_edges_identically_retained=sum(tuple(e) in candidates for e in edges),same_column_membership_count=sum(a==b for a,b in zip(edges,physical['edges'])),matched_edge_count=True,matched_edge_cardinalities=True,matched_node_degrees=True,matched_coverage=True)
 shared=dict(dataset=dataset,fit_range=[0,end],fit_split='train',protocol_fingerprint=data.metadata['fingerprint'],training_residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest(),train_only=True,validation_rows_read=0,test_rows_read=0,construction_seed=TOPOLOGY_SEED,physical_reference=physical['metadata'])
 for variant,graph,extra in [('statistical_hypergraph',stats,report),('node_independent_memory',tensors([[i] for i in range(n)],n),dict(control='one singleton support and independent retrieval per node',capacity_match_to_physical=False))]:
  meta=graph_metadata(graph);meta.update(shared);meta.update(extra);meta['variant']=variant
  if variant=='node_independent_memory':meta['value_channels']='node trajectories retrieved independently'
  write_json(ROOT/'audit/graphs'/f'{dataset}_{variant}.json',dict(edges=graph['edges'],metadata=meta))

def graph_for(dataset,variant):
 if variant not in VARIANTS:raise ValueError('unregistered structure control '+variant)
 p=ROOT/'audit/graphs'/f'{dataset}_{variant}.json';d=json.loads(p.read_text());g=tensors(d['edges'],d['metadata']['num_nodes'],d['metadata'])
 if graph_metadata(g)['incidence_sha256']!=d['metadata']['incidence_sha256']:raise RuntimeError('incidence mismatch')
 return g

def memory_storage(bank):
 # Report unique backing storages, avoiding views counted twice.
 unique={};logical={}
 for name,value in vars(bank).items():
  if torch.is_tensor(value):
   logical[name]=dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
   storage=value.untyped_storage();unique[(str(value.device),storage.data_ptr())]=storage.nbytes()
 return dict(unique_tensor_storage_bytes=sum(unique.values()),tensors=logical,excludes_adapter_and_incidence=True)

def main():
 from protocol import load
 from frozen_physics import periodic_prior
 assert not torch.cuda.is_available(),'graph preparation must not use a GPU'
 for ds in ('PEMS08','PEMS07'):
  data=load(ds);daily,weekly=periodic_prior(np.array(data.values[:data.metadata['train_end']],copy=True),288,2016);prepare(ds,data,daily,weekly)
  print(json.dumps(dict(dataset=ds,prepared=list(VARIANTS),train_end=data.metadata['train_end'])),flush=True)

if __name__=='__main__':main()
