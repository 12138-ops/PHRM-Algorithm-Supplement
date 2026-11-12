import json,os,time,subprocess,sys
from common import ROOT,write_json,log
from dispatch import call
jobs=json.loads((ROOT/'audit/input_manifest.json').read_text())['jobs']
ids=[j['id'] for j in jobs if j['dataset']=='PEMS07']
while True:
    records=[ROOT/'queue_status'/(key+'.json') for key in ids]
    if all(p.exists() and json.loads(p.read_text()).get('status') in ('COMPLETE','FAILED') for p in records):break
    time.sleep(20)
for name in ('stid','stgformer'):
    for mode in ('base','phrm'):
        call(3,f'efficiency_LargeSTSD_{name}_{mode}_inputfix',['largest_efficiency','--backbone',name,'--mode',mode],True)
log('EFFICIENCY_INPUTFIX_QUEUE_COMPLETE')
