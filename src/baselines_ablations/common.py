from pathlib import Path
import os
import hashlib,json,os,random,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECTS=Path(os.environ.get('PHRM_PROJECTS_ROOT', '.'))
SEEDS=(2022,2023,2024)
DATASETS=('PEMS03','PEMS04','PEMS07','PEMS08','Solar','Electricity','Weather')
SPECS={
 'PEMS03':dict(nodes=358,cycle=288,source=PROJECTS/'PhysicsResidualKV_MultiDataset/dataset/raw/PEMS03.npz'),
 'PEMS04':dict(nodes=307,cycle=288,source=PROJECTS/'TQNet_clean/dataset/PEMS04.npz'),
 'PEMS07':dict(nodes=883,cycle=288,source=PROJECTS/'PhysicsResidualKV_MultiDataset/dataset/raw/PEMS07.npz'),
 'PEMS08':dict(nodes=170,cycle=288,source=PROJECTS/'tqnet_physics_residual_kv_pems08/dataset/PEMS08_data.npz'),
 'Solar':dict(nodes=137,cycle=144,source=PROJECTS/'TQNet_clean/dataset/solar_AL.txt'),
 'Electricity':dict(nodes=321,cycle=168,source=PROJECTS/'TQNet_clean/dataset/electricity_OT.csv'),
 'Weather':dict(nodes=21,cycle=144,source=PROJECTS/'weather_tqnet_kv_20261015/dataset/weather_wide.csv'),
}
ARCH_FIELDS=('batch_size','learning_rate','train_epochs','patience','lradj','pct_start','seq_len','pred_len','enc_in','cycle','features','d_model','d_ff','e_layers','n_heads','dropout','activation','factor','output_attention','embed','freq','fc_dropout','head_dropout','patch_len','stride','padding_patch','revin','affine','subtract_last','decomposition','kernel_size','individual','use_revin','loss','label_len')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def write_json(path,obj):
 p=Path(path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('writes must stay inside the new batch root')
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def require_gpu():
 g=os.environ.get('CUDA_VISIBLE_DEVICES')
 if g not in ('1','2','3'):raise RuntimeError('only explicitly isolated physical GPU1/2/3 are permitted')
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
 return int(g),torch.device('cuda:0')

def log(event,**values):print(json.dumps(dict(event=event,time=time.time(),**values),ensure_ascii=False),flush=True)

def old_records():return json.loads((ROOT/'audit/old_baselines.json').read_text())

def recipe_fields(args):return {k:args[k] for k in ARCH_FIELDS if k in args}

def checkpoint_for_record(record):
 return PROJECTS/'TQNet_clean/checkpoints'/record['test_settings'][-1]/'checkpoint.pth'

def result_path(job):return ROOT/'results'/job['id']/'result.json'
