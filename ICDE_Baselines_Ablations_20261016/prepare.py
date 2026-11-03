"""Prepare immutable inputs and the complete predeclared job graph."""
import argparse,json,re,subprocess
from pathlib import Path
from common import *
from protocol import prepare
from ablation_model import VARIANTS

def gtr_recipes():
 defaults=dict(seq_len=96,pred_len=96,enc_in=7,cycle=24,model_type='mlp',d_model=512,dropout=0,use_revin=1,individual=0,batch_size=128,learning_rate=.0001,train_epochs=30,patience=5,lradj='type3',pct_start=.3,features='M')
 out={}
 for ds in DATASETS:
  script=ROOT/'vendor/GTR/scripts/GTR'/f'{ds.lower()}.sh';text=script.read_text();args=defaults.copy()
  for k in defaults:
   m=re.search(r'--'+k+r'\s+([^\s\\]+)',text)
   if m:
    try:args[k]=json.loads(m[1])
    except json.JSONDecodeError:args[k]=m[1]
  args['enc_in']=SPECS[ds]['nodes'];args['cycle']=SPECS[ds]['cycle'];args['seq_len']=96;args['pred_len']=96
  out[ds]=args
 return out

def jobs():
 rows=[]
 for model in ('PatchTST','iTransformer'):
  for h in (12,24,48,96):rows.append(dict(id=f'select_PEMS08_{model}_h{h}',kind='selection',dataset='PEMS08',model=model,horizon=h,seed=2024,priority=0,preferred_gpu=2,dependencies=[]))
 for ds in DATASETS:
  hs=(12,24,48,96) if ds.startswith('PEMS') else (96,192,336,720)
  for model in ('PatchTST','iTransformer','GTR'):
   for h in hs:
    for seed in SEEDS:
     dependencies=[f'select_PEMS08_{model}_h{h}'] if ds=='PEMS08' and model!='GTR' else []
     rows.append(dict(id=f'baseline_{ds}_{model}_h{h}_s{seed}',kind='baseline',dataset=ds,model=model,horizon=h,seed=seed,
       priority=10+(0 if model=='GTR' else 4)+(0 if seed==2024 else 2),preferred_gpu=3 if ds=='PEMS07' else 1 if ds in ('Solar','Electricity','Weather') else 2,dependencies=dependencies))
 for ds,h in [('PEMS08',96),('PEMS07',96),('PEMS08',24)]:
  for variant in VARIANTS:
   for seed in SEEDS:
    rows.append(dict(id=f'ablation_{ds}_{variant}_h{h}_s{seed}',kind='ablation',dataset=ds,horizon=h,variant=variant,seed=seed,priority=3 if h==96 and seed==2024 else 7 if h==96 else 22,preferred_gpu=3 if ds=='PEMS07' else 2,dependencies=[]))
 for ds in ('PEMS08','PEMS07'):
  for label,change in [('memory_half',dict(memory_stride=24)),('memory_double',dict(memory_stride=6)),('topk4',dict(topk=4)),('topk16',dict(topk=16))]:
   rows.append(dict(id=f'sensitivity_{ds}_{label}_h96_s2024',kind='ablation',dataset=ds,horizon=96,variant='full',seed=2024,validation_only=True,priority=24,preferred_gpu=3 if ds=='PEMS07' else 2,dependencies=[],**change))
 assert len({r['id'] for r in rows})==len(rows)
 assert all(r['dataset']!='Traffic' for r in rows)
 return rows

def main():
 for ds in DATASETS:prepare(ds)
 write_json(ROOT/'audit/gtr_recipes.json',gtr_recipes())
 manifest=dict(version='ICDE-bulk-baseline-ablation-v1',datasets=list(DATASETS),retained_standard_spatiotemporal_datasets=['METR-LA','LargeST-SD'],excluded=['Traffic','ETTh1','ETTh2','ETTm1','ETTm2'],physical_gpus=[1,2,3],seeds=list(SEEDS),jobs=jobs(),
  predeclared_selection='PEMS08 old recipe by full-validation MSE using seed2024 only; no test-driven choice',
  comparison_note='new ablation full controls define the matched ablation reference; protected historical main results remain unchanged',
  pairwise_budget_note='all genuine roads, with actual edge counts disclosed; no fictitious edges to force equal counts',
  sensitivity_split='validation only, predeclared sweeps; never use test for model selection')
 write_json(ROOT/'audit/jobs.json',manifest)
 # Materialize already-audited exact-configuration reuse without a GPU job.
 from baselines import recipe,compatible_reuse,run
 reused=0
 for job in manifest['jobs']:
  if job['kind']!='baseline' or job['model']=='GTR' or job['dataset']=='PEMS08':continue
  a=recipe(job['dataset'],job['model'],job['horizon'])
  if compatible_reuse(job['dataset'],job['model'],job['horizon'],job['seed'],a):run(job,None);reused+=1
 write_json(ROOT/'audit/preparation.json',dict(prepared=True,total_jobs=len(manifest['jobs']),reused_baseline_cells=reused,code_frozen=False))
 log('BATCH_PREPARED',total_jobs=len(manifest['jobs']),reused=reused)

if __name__=='__main__':main()
