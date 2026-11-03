"""Reassign PEMS07 to GPU2 after the PEMS08 queue drains if GPU3 stays busy."""
import json, os, subprocess, sys, time
from pathlib import Path
from common import ROOT, write_json, log
from three_dispatch import call

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('fallback manager must be CPU-only')
jobs=json.loads((ROOT/'audit/jobs.json').read_text())['jobs']
while True:
    p8=[ROOT/'queue_status'/(j['id']+'.json') for j in jobs if j['dataset']=='PEMS08']
    if all(p.exists() and json.loads(p.read_text()).get('status')=='COMPLETE' for p in p8):
        break
    time.sleep(30)
log('PEMS08_DRAINED_FALLBACK_TO_GPU2')
for j in jobs:
    if j['dataset']!='PEMS07':
        continue
    call(2,j['id'],['fit','--index',str(jobs.index(j))])
log('PEMS07_GPU2_FALLBACK_COMPLETE')
