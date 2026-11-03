import argparse,json,traceback,time
from common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();gpu,device=require_gpu()
 frozen=json.loads((ROOT/'audit/code_manifest.json').read_text())
 for rel,d in frozen['files'].items():
  if sha(ROOT/rel)!=d:raise RuntimeError('frozen batch code/input changed: '+rel)
 jobs=json.loads((ROOT/'audit/jobs.json').read_text())['jobs'];job=next(j for j in jobs if j['id']==a.job)
 if result_path(job).exists():log('SKIP_COMPLETE',job=job['id']);return
 log('JOB_START',job=job,physical_gpu=gpu)
 if job['kind']=='selection':
  from baselines import select_recipe
  out=select_recipe(job,device);write_json(result_path(job),dict(status='COMPLETE',job=job,selection=out,test_evaluated=False))
 elif job['kind']=='baseline':
  from baselines import run
  run(job,device)
 elif job['kind']=='ablation':
  from ablations import run
  run(job,device)
 else:raise ValueError(job['kind'])
 log('JOB_COMPLETE',job=job['id'],physical_gpu=gpu)

if __name__=='__main__':main()
