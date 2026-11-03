"""Incidence-margin preserving, metadata-only local interventions."""
import csv,json,math,random,time
from collections import Counter
import numpy as np
from common import ROOT,STRUCTURE_SEEDS,write_json,sha,digest,arrays

def distance_km(records):
 lat=np.deg2rad(np.array([float(r['Lat']) for r in records]));lon=np.deg2rad(np.array([float(r['Lng']) for r in records]))
 a=np.sin((lat[:,None]-lat)/2)**2+np.cos(lat[:,None])*np.cos(lat)[None,:]*np.sin((lon[:,None]-lon)/2)**2
 return 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def margins(edges,n):
 degree=np.zeros(n,dtype=np.int64)
 for e in edges:degree[e]+=1
 return [len(e) for e in edges],degree

def all_distances(edges,d):
 values=[]
 for e in edges:
  if len(e)>1:
   a=d[np.ix_(e,e)];values.extend(a[np.triu_indices(len(e),1)].tolist())
 return np.asarray(values)

def metadata(edges,physical,d,records):
 sizes,degrees=margins(edges,len(records));s0,d0=margins(physical,len(records))
 distances=all_distances(edges,d);original=all_distances(physical,d)
 shared=sum(len(set(e)&set(p)) for e,p in zip(edges,physical)) if len(edges)==len(physical) else None
 labels=[(r['Fwy'].strip(),r['Direction'].strip()) for r in records]
 coherent=[len({labels[i] for i in e})==1 for e in edges if len(e)>1]
 result=dict(nodes=len(records),supports=len(edges),higher_order=sum(s>2 for s in sizes),singletons=sum(s==1 for s in sizes),
  covered_nodes=int(np.count_nonzero(degrees)),edge_sizes=sizes,node_degrees=degrees.tolist(),edges_sha256=digest(edges),
  matched_edge_count=len(edges)==len(physical),matched_ordered_sizes=sizes==s0,matched_node_degrees=bool(np.array_equal(degrees,d0)),
  mean_pairwise_geodesic_km=float(distances.mean()) if len(distances) else None,
  distance_quantiles_km=np.quantile(distances,[.1,.25,.5,.75,.9]).tolist() if len(distances) else None,
  physical_mean_pairwise_geodesic_km=float(original.mean()),same_highway_direction_support_fraction=float(np.mean(coherent)) if coherent else None,
  changed_original_membership_fraction=1-shared/sum(s0) if shared is not None else None,
  changed_incidence_entries_fraction=2*(sum(s0)-shared)/(len(records)*len(physical)) if shared is not None else None,
  changed_supports=sum(set(e)!=set(p) for e,p in zip(edges,physical)) if shared is not None else None)
 return result

def local_chain(physical,d,records,seed,attempt_factor=10):
 """Swap geographically close, differently labelled nodes across two supports.

 Only metadata is used. Each support retains its original 500m bin of mean
 pairwise geodesic distance. Singleton identities are frozen. p denotes the
 attempt-budget fraction, not a guaranteed fraction of memberships changed.
 """
 n=len(records);edges=[set(e) for e in physical];seen={tuple(sorted(e)) for e in edges}
 labels=[(r['Fwy'].strip(),r['Direction'].strip()) for r in records]
 active=[i for i,e in enumerate(edges) if len(e)>1]
 incident=[set() for _ in range(n)]
 for j in active:
  for a in edges[j]:incident[a].add(j)
 near={a:[b for b in range(n) if 0<d[a,b]<=1.0 and labels[a]!=labels[b] and incident[b]] for a in range(n) if incident[a]}
 eligible=sorted(a for a,b in near.items() if b)
 if not eligible:raise RuntimeError('No local cross-road/direction swaps available')
 rng=random.Random(seed);initial_bins=[];pair_sums=[]
 for e in physical:
  if len(e)>1:
   total=float(d[np.ix_(e,e)].sum()/2);pair_sums.append(total);initial_bins.append(int((total/math.comb(len(e),2))/.5))
  else:pair_sums.append(0.);initial_bins.append(-1)
 total_attempts=attempt_factor*sum(map(len,physical));marks={int(total_attempts*p):p for p in [.25,.5,1.]}
 accepted=0;snapshots={}
 for attempt in range(1,total_attempts+1):
  a=rng.choice(eligible);b=rng.choice(near[a]);i=rng.choice(sorted(incident[a]));j=rng.choice(sorted(incident[b]))
  if i!=j and b not in edges[i] and a not in edges[j]:
   left=sorted(edges[i]-{a});right=sorted(edges[j]-{b})
   si=pair_sums[i]+float(d[b,left].sum()-d[a,left].sum());sj=pair_sums[j]+float(d[a,right].sum()-d[b,right].sum())
   if int(max(si,0)/math.comb(len(edges[i]),2)/.5)==initial_bins[i] and int(max(sj,0)/math.comb(len(edges[j]),2)/.5)==initial_bins[j]:
    ei=tuple(sorted(left+[b]));ej=tuple(sorted(right+[a]))
    if ei!=ej and ei not in seen and ej not in seen:
     seen.remove(tuple(sorted(edges[i])));seen.remove(tuple(sorted(edges[j])));seen.update((ei,ej))
     edges[i]=set(ei);edges[j]=set(ej);pair_sums[i]=si;pair_sums[j]=sj
     incident[a].remove(i);incident[a].add(j);incident[b].remove(j);incident[b].add(i);accepted+=1
  if attempt in marks:
   final=[sorted(e) for e in edges];p=marks[attempt];meta=metadata(final,physical,d,records)
   assert meta['matched_ordered_sizes'] and meta['matched_node_degrees'] and meta['covered_nodes']==n
   assert all(e==q for e,q in zip(final,physical) if len(q)==1)
   meta.update(structure_seed=seed,attempts=attempt,accepted_swaps=accepted,perturbation_budget_fraction=p,
    replacement_distance_limit_km=1.,mean_support_distance_bin_km=.5,
    support_mean_distance_bins_preserved=True,uses_observation_values=False,
    interpretation='geographically local, differently labelled sensor swaps; distance bins and incidence margins fixed; actual changes reported')
   snapshots[p]=(final,meta)
 if accepted<100:raise RuntimeError('Too few feasible local swaps; do not label this a randomized control')
 return snapshots

def correlated_graph(physical,c,seed=19073,attempts=200000):
 """Training-only signed-correlation optimization with fixed incidence margins."""
 rng=random.Random(seed);edges=[set(e) for e in physical];seen={tuple(sorted(e)) for e in edges};active=[i for i,e in enumerate(edges) if len(e)>1]
 def swap(i,j,a,b,improve):
  if a==b or b in edges[i] or a in edges[j]:return False
  l=sorted(edges[i]-{a});r=sorted(edges[j]-{b})
  if improve and ((c[b,l].sum()-c[a,l].sum())/math.comb(len(edges[i]),2)+(c[a,r].sum()-c[b,r].sum())/math.comb(len(edges[j]),2))<=1e-12:return False
  ei=tuple(sorted(l+[b]));ej=tuple(sorted(r+[a]))
  if ei==ej or ei in seen or ej in seen:return False
  seen.remove(tuple(sorted(edges[i])));seen.remove(tuple(sorted(edges[j])));seen.update((ei,ej));edges[i]=set(ei);edges[j]=set(ej);return True
 for _ in range(10*sum(map(len,physical))):
  i,j=rng.sample(active,2);swap(i,j,rng.choice(sorted(edges[i])),rng.choice(sorted(edges[j])),False)
 accepted=0
 for _ in range(attempts):
  i,j=rng.sample(active,2);accepted+=swap(i,j,rng.choice(sorted(edges[i])),rng.choice(sorted(edges[j])),True)
 return [sorted(e) for e in edges],dict(optimization_attempts=attempts,accepted_improvements=accepted,structure_seed=seed,
  objective='equal-weight mean signed Pearson correlation on training periodic residuals',fit_split='train',validation_rows_read=0,test_rows_read=0)

def tensors(edges,n,meta=None):
 import torch
 a=torch.zeros(n,len(edges),dtype=torch.float32)
 for j,e in enumerate(edges):a[e,j]=1
 assert bool((a.sum(1)>0).all())
 return dict(incidence=a,pool=a/a.sum(1,keepdim=True),edge_pool=a/a.sum(0,keepdim=True),edges=edges,num_nodes=n,metadata=meta or {})

def prepare():
 import torch
 from physical_graph import load_graph
 from physics_kv import periodic_prior
 from memory_sampling import select_memory_origins
 torch.set_num_threads(2)
 a=arrays();g=load_graph(ROOT/'data/physical/sd_rn_adj.npy',ROOT/'data/physical/sd_meta.csv',expected_sensor_ids=a['node_ids'])
 physical=g['edges'];n=a['data'].shape[1]
 with (ROOT/'data/physical/sd_meta.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
 assert [str(x) for x in a['node_ids']]==[r['ID'] for r in records]
 d=distance_km(records);cutoff=int(a['origins_val'][0])-13;assert cutoff==21009
 daily,weekly=periodic_prior(a['data'][:cutoff],96,672)
 residual=a['data'][:cutoff]-daily.numpy()[np.arange(cutoff)%96]-weekly.numpy()[np.arange(cutoff)%672]
 cov=np.cov(residual.astype(np.float64),rowvar=False,bias=True);sd=np.sqrt(np.diag(cov));corr=cov/np.maximum(sd[:,None]*sd[None,:],1e-12);np.fill_diagonal(corr,0)
 np.save(ROOT/'audit/train_residual_covariance.npy',cov);np.save(ROOT/'audit/geodesic_distance_km.npy',d)
 graphs={}
 graphs['physical']=(physical,metadata(physical,physical,d,records)|dict(original_physical_metadata=g['metadata'],edge_provenance=g['edge_metadata']))
 for seed in STRUCTURE_SEEDS:
  chain=local_chain(physical,d,records,seed)
  for p,(edges,meta) in chain.items():graphs[f'local_p{int(p*100):03d}_g{seed}']=(edges,meta)
 edges,meta=correlated_graph(physical,corr);graphs['correlation']=(edges,metadata(edges,physical,d,records)|meta)
 adj=np.load(ROOT/'data/physical/sd_rn_adj.npy');i,j=np.where(np.triu(np.maximum(adj,adj.T)>0,1))
 pairs=[[int(x),int(y)] for x,y in zip(i,j)];covered=set(sum(pairs,[]));pairs.extend([[k] for k in range(n) if k not in covered])
 graphs['pairwise']=(pairs,metadata(pairs,physical,d,records));graphs['singleton']=([[i] for i in range(n)],metadata([[i] for i in range(n)],physical,d,records))
 for name,(edges,meta) in graphs.items():
  coherences=[]
  for e in edges:
   den=float(np.diag(cov)[e].sum());coherences.append(float(cov[np.ix_(e,e)].sum()/(len(e)*den)) if den>0 else 0.)
  meta.update(training_residual_coherence=coherences,coherence_formula='|e| Var(mean residual over e) / sum_n Var(residual_n)',
              coherence_fit_rows=[0,cutoff],coherence_singletons_excluded_from_comparisons=True)
  write_json(ROOT/f'graphs/{name}.json',dict(name=name,edges=edges,metadata=meta))
 jobs=[dict(id='physical_reference',kind='reuse',graph='physical',seed=2024)]
 for p in [1.,.25,.5]:
  for seed in STRUCTURE_SEEDS:
   name=f'local_p{int(p*100):03d}_g{seed}';jobs.append(dict(id=name,kind='fit',graph=name,seed=2024))
 jobs.extend(dict(id=name,kind='fit',graph=name,seed=2024) for name in ['correlation','pairwise','singleton'])
 assert sum(j['kind']=='fit' for j in jobs)==12
 write_json(ROOT/'audit/jobs.json',jobs)
 write_json(ROOT/'audit/graph_preflight.json',dict(graphs=len(graphs),new_fits=12,reused_reference=1,train_end=cutoff,
    original_supports=len(physical),all_supports_covered=True,test_evaluated=False,
    perturbation_x_axis='Actual removed-original-membership fraction, also store requested swap-budget fraction',
    physical_budget_episodes=1749,geometry_used='Official sensor lat/lon; haversine geodesic distance, not sensor-index distance'))
 print(json.dumps(dict(stage='GRAPHS_PREPARED',graphs=len(graphs),fits=12,physical_supports=len(physical))),flush=True)

if __name__=='__main__':prepare()
