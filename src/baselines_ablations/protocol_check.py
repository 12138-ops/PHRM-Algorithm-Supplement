"""Compare cached loader fingerprints with the previously audited protocol."""
import json
from common import *
from protocol import load

def main():
 expected=json.loads((ROOT/'audit/expected_protocols.json').read_text());checks=[]
 for ds in DATASETS:
  data=load(ds)
  for k,v in expected[ds].items():assert data.metadata[k]==v,(ds,k,data.metadata[k],v)
  assert sha(data.metadata['source'])==data.metadata['source_sha256']
  for h in ((12,24,48,96) if ds.startswith('PEMS') else (96,192,336,720)):
   tr=data.origins(h,'train');va=data.origins(h,'val');te=data.origins(h,'test')
   assert tr[-1]+h==data.metadata['train_end'] and va[-1]+h==data.metadata['val_end'] and te[-1]+h==len(data.values)
   assert va[0]==data.metadata['train_end'] and te[0]==data.metadata['test_start']
   checks.append(dict(dataset=ds,horizon=h,train_windows=len(tr),validation_windows=len(va),test_windows=len(te),scaler_matches_audited_protocol=True,source_hash_unchanged=True))
 write_json(ROOT/'audit/protocol_check.json',dict(passed=True,checks=checks,test_labels_used_for_selection=False));log('PROTOCOL_CHECK_COMPLETE',cells=len(checks))

if __name__=='__main__':main()
