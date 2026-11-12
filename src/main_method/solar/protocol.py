"""Data and window protocol matched to the released TQNet loaders."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from settings import OFFICIAL, PROJECTS, SEEDS, SPECS, checkpoint_path

PROTOCOL_VERSION = 'tqnet-standardscaler-official-borders-v2'


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_raw(dataset, path):
    path=Path(path)
    if dataset.startswith('PEMS'):
        with np.load(path,allow_pickle=False) as f: raw=f['data'][:,:,0]
    elif dataset=='Solar':
        raw=np.loadtxt(path,delimiter=',',dtype=np.float64)
    elif dataset=='Electricity' or dataset.startswith('ETT'):
        frame=pd.read_csv(path)
        cols=[c for c in frame.columns if c not in ('date','OT')]
        raw=frame[cols+['OT']].values
        if dataset.startswith('ETT'):
            assert cols+['OT']==['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
            dates=pd.to_datetime(frame['date']); minutes=15 if dataset.startswith('ETTm') else 60
            assert (dates.diff().iloc[1:]==pd.Timedelta(minutes=minutes)).all()
    else: raise ValueError(dataset)
    if raw.ndim!=2 or raw.shape[1]!=SPECS[dataset]['nodes'] or not np.isfinite(raw).all():
        raise ValueError(f'invalid {dataset} data shape or non-finite values: {raw.shape}')
    return raw


def borders(dataset,n):
    if dataset.startswith('ETT'):
        factor=4 if dataset.startswith('ETTm') else 1
        assert n>=20*30*24*factor
        return 12*30*24*factor,16*30*24*factor,16*30*24*factor
    if dataset.startswith('PEMS'):
        train_end=int(n*.6); val_end=train_end+int(n*.2)
    elif dataset=='Solar':
        train_end=int(n*.7); val_end=train_end+int(n*.1)
    elif dataset=='Electricity':
        train_end=int(n*.7); val_end=n-int(n*.2)
    else: raise ValueError(dataset)
    return train_end,val_end,n-int(n*.2)


@dataclass
class ProtocolData:
    dataset: str
    data: np.ndarray
    scaler: StandardScaler
    metadata: dict

    @property
    def train_end(self): return self.metadata['train_end']
    @property
    def val_end(self): return self.metadata['val_end']

    def origins(self,horizon):
        n=len(self.data); h=int(horizon)
        train=np.arange(96,self.train_end-h+1,dtype=np.int64)
        val=np.arange(self.train_end,self.val_end-h+1,dtype=np.int64)
        test=np.arange(self.metadata['test_start'],n-h+1,dtype=np.int64)
        assert all(len(x)>0 for x in (train,val,test))
        assert train[-1]+h<=self.train_end and val[-1]+h<=self.val_end and test[-1]+h<=n
        return train,val,test

    def save_scaler(self,root):
        out=Path(root)/'protocol_artifacts'/self.dataset
        out.mkdir(parents=True,exist_ok=True)
        # Only the corrected project receives these outputs.
        np.savez(out/'scaler.npz',mean=self.scaler.mean_,scale=self.scaler.scale_,var=self.scaler.var_)
        (out/'protocol.json').write_text(json.dumps(self.metadata,indent=2))


def load_data(dataset,source=None):
    spec=SPECS[dataset]; source=Path(source or spec['source'])
    raw=read_raw(dataset,source); n=len(raw)
    train_end,val_end,test_start=borders(dataset,n)
    scaler=StandardScaler().fit(raw[:train_end])
    data=scaler.transform(raw).astype(np.float32)
    if dataset.startswith('ETT'): data=data[:20*30*24*(4 if dataset.startswith('ETTm') else 1)]
    scaler_hash=hashlib.sha256(scaler.mean_.tobytes()+scaler.scale_.tobytes()).hexdigest()
    metadata=dict(version=PROTOCOL_VERSION,dataset=dataset,protocol=spec['split'],
                  source=source.name,source_sha256=sha256_file(source),shape=list(raw.shape),
                  seq_len=96,train_end=train_end,val_end=val_end,test_start=test_start,
                  scaler='sklearn.preprocessing.StandardScaler',scaler_fit_range=[0,train_end],
                  scaler_sha256=scaler_hash,metric_space='per_channel_train_standardized',
                  base_cycle=spec['cycle'],daily_period=spec['daily'],weekly_period=spec['weekly'],
                  use_revin=spec['use_revin'],base_dropout=spec.get('dropout',0),evaluation_end=len(data),
                  topology=spec['topology'],cycle_index='forecast_origin modulo base_cycle')
    metadata['fingerprint']=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    if not np.isfinite(data).all(): raise ValueError('non-finite standardized data')
    return ProtocolData(dataset,data,scaler,metadata)
