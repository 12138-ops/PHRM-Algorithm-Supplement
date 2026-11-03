"""Validation-only physical-support study; all writes stay in this project."""
from pathlib import Path
import hashlib,json,os,random
import numpy as np

ROOT=Path(__file__).resolve().parent
STRUCTURE_SEEDS=(19073,29073,39073)
LENGTH=HORIZON=12
PERIOD=96
MODEL_SEED=2024
GPU_IDS=(1,2,3)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_json(path,value):
 p=Path(path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('output outside isolated project')
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');t.replace(p)

def check_protected():
 d=json.loads((ROOT/'audit/protected_inputs.json').read_text())
 for p,h in d.items():
  if sha(p)!=h:raise RuntimeError('Frozen input changed: '+p)
 manifest=ROOT/'audit/code_manifest.json'
 if manifest.exists():
  for name,h in json.loads(manifest.read_text())['files'].items():
   if sha(ROOT/name)!=h:raise RuntimeError('Frozen study code/graph changed: '+name)
 return len(d)

def arrays():
 with np.load(ROOT/'data/processed/LargeSTSD_2019_15min_l12_h12.npz',allow_pickle=False) as z:
  d={k:z[k] for k in z.files}
 assert 'origins_test' not in d and d['data'].shape==(28036,716)
 assert d['raw'].shape==d['data'].shape
 assert len(d['origins_train'])==21010 and len(d['origins_val'])==7003
 assert d['origins_train'][0]==12 and d['origins_val'][0]==21022
 assert d['origins_val'][-1]+HORIZON==len(d['data'])
 assert d['data'].dtype==np.float32 and np.isfinite(d['data']).all()
 return d

def require_gpu():
 s=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if s not in tuple(map(str,GPU_IDS)):raise RuntimeError('Only one physical GPU1/2/3 may be visible')
 import torch
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One CUDA device required')
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 return int(s),torch.device('cuda:0')

def seed_all():
 import torch
 random.seed(MODEL_SEED);np.random.seed(MODEL_SEED);torch.manual_seed(MODEL_SEED);torch.cuda.manual_seed_all(MODEL_SEED)
