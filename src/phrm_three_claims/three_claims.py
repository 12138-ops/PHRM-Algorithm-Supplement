import argparse,csv,json,time
from pathlib import Path
import numpy as np, torch
from common import ROOT,write_json,log,require_gpu,sha
from ablations import setup
from ablation_model import build_bank,graph_for
from structure_controls import memory_storage
from evidence import main_job

def summary():
    rows=[]
    for p in (ROOT/'results').glob('*/result.json'):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get('status')!='COMPLETE':continue
        rows.append(dict(dataset=r['dataset'],horizon=r['horizon'],seed=r['seed'],variant=r['variant'],mse=r['validation']['trained']['mse'],mae=r['validation']['trained']['mae'],params=r.get('trainable_parameters'),storage_bytes=r.get('memory_storage',{}).get('unique_tensor_storage_bytes'),episodes=r.get('memory_episodes'),supports=r.get('graph',{}).get('edges'),covered_nodes=r.get('graph',{}).get('covered_nodes'),train_only=r.get('memory_train_only'),test_labels_used_for_selection=r.get('test_labels_used_for_selection')))
    write_json(ROOT/'results'/'memory_only_rows.json',rows)
    out={}
    for ds in ('PEMS08','PEMS07'):
      out[ds]={}
      for v in ('periodic_only','node_level_memory','random_support','physical_support','support_shuffled'):
        a=[x for x in rows if x['dataset']==ds and x['variant']==v]
        if not a:continue
        out[ds][v]=dict(n=len(a),mse_mean=float(np.mean([x['mse'] for x in a])),mse_std=float(np.std([x['mse'] for x in a],ddof=1)) if len(a)>1 else 0.,mae_mean=float(np.mean([x['mae'] for x in a])),mae_std=float(np.std([x['mae'] for x in a],ddof=1)) if len(a)>1 else 0.,parameter_counts=sorted(set(x['params'] for x in a)),storage_bytes=sorted(set(x['storage_bytes'] for x in a)),support_counts=sorted(set(x['supports'] for x in a)))
    checks=dict(rows=len(rows),all_validation_only=all(x['train_only'] and not x['test_labels_used_for_selection'] for x in rows),provenance_fields=['support membership','memory origin','retrieval weights'],provenance_quality='computed after fits from chronology and support coverage; no test labels')
    write_json(ROOT/'results'/'memory_only_summary.json',dict(groups=out,checks=checks,created=time.time()))
    log('MEMORY_SUMMARY_COMPLETE',rows=len(rows),groups=out)

def scalability(device):
    records=[]
    for ds in ('PEMS08','PEMS07'):
      job=main_job(ds);data,model,bank,graph,ck,bank_o=setup(job,device);h=96;batch=16 if ds=='PEMS08' else 4
      counts=[]
      for c in (64,128,256,512,1024):
        if c>len(bank_o):continue
        oo=bank_o[:c];torch.cuda.synchronize();t0=time.perf_counter();b=build_bank(data,oo,h,graph,model.daily,model.weekly,device,batch,'physical_support',topk=8);torch.cuda.synchronize();build=time.perf_counter()-t0
        x=torch.from_numpy(np.stack([data.values[t-96:t] for t in data.origins(h,'val')[:batch]])).to(device);o=torch.as_tensor(data.origins(h,'val')[:batch],device=device);w=torch.ones(h,device=device)
        for _ in range(5):model(x,o,b,w)
        torch.cuda.synchronize();times=[]
        for _ in range(15):
          t=time.perf_counter();model(x,o,b,w);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
        counts.append(dict(memory_episodes=c,build_seconds=build,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),storage=memory_storage(b)))
      records.append(dict(dataset=ds,nodes=data.values.shape[1],physical_supports=graph['incidence'].shape[1],memory_counts=counts,exact_retrieval=True,ann_backend='not run; exact torch distance retained as reference',train_only_bank=True,validation_inputs=True))
    write_json(ROOT/'results'/'scalability_summary.json',dict(records=records,created=time.time()))
    log('SCALABILITY_COMPLETE',datasets=[x['dataset'] for x in records])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('task',choices=['summary','scalability']);a=p.parse_args()
 if a.task=='summary':summary()
 else:
  _,device=require_gpu();torch.set_num_threads(2);scalability(device)
