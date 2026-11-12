from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
from common import ROOT,SPECS,sha,digest,write_json,log

@dataclass
class Data:
 values:np.ndarray
 marks:np.ndarray
 metadata:dict
 def origins(self,h,split):
  m=self.metadata;n=m['shape'][0]
  first,last={'train':(96,m['train_end']-h),'val':(m['train_end'],m['val_end']-h),'test':(m['test_start'],n-h)}[split]
  return np.arange(first,last+1,dtype=np.int64)

def prepare(dataset):
 spec=SPECS[dataset];p=Path(spec['source']);source_hash=sha(p)
 if dataset.startswith('PEMS'):
  with np.load(p,allow_pickle=False) as z:raw=z['data'][:,:,0]
  marks=np.zeros((len(raw),1),dtype='float32');names=list(map(str,range(raw.shape[1])))
 elif dataset=='Solar':
  raw=np.loadtxt(p,delimiter=',',dtype=np.float64);marks=np.zeros((len(raw),1),dtype='float32');names=list(map(str,range(raw.shape[1])))
 else:
  frame=pd.read_csv(p);names=[c for c in frame if c not in ('date','OT')]+['OT'];raw=frame[names].values
  from utils.timefeatures import time_features
  # The old aligned baseline logs use freq=h, including Weather. Preserve the
  # model's historical covariate convention; GTR does not consume these marks.
  marks=time_features(pd.to_datetime(frame['date'].values),freq='h').T.astype('float32')
 n=len(raw)
 if dataset.startswith('PEMS'):tr=int(n*.6);ve=tr+int(n*.2)
 elif dataset=='Solar':tr=int(n*.7);ve=tr+int(n*.1)
 else:tr=int(n*.7);ve=n-int(n*.2)
 ts=n-int(n*.2);sc=StandardScaler().fit(raw[:tr]);values=sc.transform(raw).astype('float32')
 assert raw.shape[1]==spec['nodes'] and np.isfinite(values).all()
 meta=dict(dataset=dataset,shape=list(values.shape),source=str(p),source_sha256=source_hash,seq_len=96,
           train_end=tr,val_end=ve,test_start=ts,metric_space='per_channel_train_standardized',
           scaler_fit_range=[0,tr],scaler_sha256=__import__('hashlib').sha256(sc.mean_.tobytes()+sc.scale_.tobytes()).hexdigest(),
           variables=names,calendar_covariates='legacy time_features(freq=h) for custom datasets; zeros for PEMS/Solar',
           test_drop_last=False,test_stride=1)
 meta['fingerprint']=digest(meta);out=ROOT/'data_cache'/dataset;out.mkdir(exist_ok=True,parents=True)
 for name,arr in [('data',values),('marks',marks),('mean',sc.mean_),('scale',sc.scale_)]:np.save(out/(name+'.npy'),arr,allow_pickle=False)
 write_json(out/'protocol.json',meta);log('DATA_PREPARED',dataset=dataset,shape=list(values.shape),train_end=tr,val_end=ve,test_start=ts)
 return meta

def load(dataset):
 p=ROOT/'data_cache'/dataset
 meta=json.loads((p/'protocol.json').read_text())
 assert meta['dataset']==dataset and meta['seq_len']==96
 return Data(np.load(p/'data.npy',mmap_mode='r',allow_pickle=False),np.load(p/'marks.npy',mmap_mode='r',allow_pickle=False),meta)

class Windows(Dataset):
 def __init__(self,data,origins,horizon):self.data=data;self.origins=np.asarray(origins);self.h=int(horizon)
 def __len__(self):return len(self.origins)
 def __getitem__(self,i):
  t=int(self.origins[i]);a=self.data
  return torch.from_numpy(a.values[t-96:t].copy()),torch.from_numpy(a.values[t:t+self.h].copy()),torch.tensor(t),torch.from_numpy(a.marks[t-96:t].copy())

def loader(data,h,split,batch,*,shuffle=False,drop_last=False,workers=2):
 return DataLoader(Windows(data,data.origins(h,split),h),batch_size=batch,shuffle=shuffle,drop_last=drop_last,num_workers=workers,pin_memory=True)
