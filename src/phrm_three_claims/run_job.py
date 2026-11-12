import argparse,json
from pathlib import Path
import torch
from common import ROOT,require_gpu
from ablations import run
from three_claims import summary,scalability
p=argparse.ArgumentParser();p.add_argument('task',choices=['fit','summary','scalability']);p.add_argument('--index',type=int);a=p.parse_args()
if a.task=='fit':
 j=json.loads((ROOT/'audit/jobs.json').read_text())['jobs'][a.index];assert j['validation_only'] and not j.get('test_evaluated',False);_,d=require_gpu();torch.set_num_threads(2);run(j,d)
elif a.task=='summary':summary()
else:
 _,d=require_gpu();torch.set_num_threads(2);scalability(d)
