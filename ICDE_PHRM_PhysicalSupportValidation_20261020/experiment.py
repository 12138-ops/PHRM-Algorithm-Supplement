"""Frozen L12/H12 TQNet--physical KV recipe, extended to three paired seeds."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.tqnet import Model as TQNet
from physical_graph import load_graph
from physics_kv import (TQNetPhysicsKV, periodic_prior, residual_history,
                        residual_future, edge_descriptors, trainable_parameters)
from memory_sampling import (select_memory_origins, sampling_metadata,
                             build_phase_balanced_bank)
from metrics import Metrics, masked_mae_loss

LENGTH = HORIZON = 12
PERIOD = 96
EVAL_BATCH = 64
VERSION = 'largest-sd-official-l12-h12-paired-validation-v1'
SEEDS = (2022, 2023, 2024)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def stage(name, **kwargs):
    print('LARGEST_STAGE '+json.dumps(dict(stage=name,time=time.time(),**kwargs)),flush=True)


def config(nodes=716):
    return SimpleNamespace(seq_len=LENGTH,pred_len=HORIZON,enc_in=nodes,cycle=PERIOD,
                           model_type='mlp',d_model=512,dropout=0.,use_revin=0)


def prepared(root, include_test=False):
    path=root/'data/processed/LargeSTSD_2019_15min_l12_h12.npz'
    with np.load(path,allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    data=arrays['data'];raw=arrays['raw']
    assert data.shape==raw.shape==(35040,716)
    assert np.isfinite(data).all() and np.isfinite(raw).all()
    # The released indices point to the last observed row. The KV adapter uses
    # forecast origins (first target row), so convert once, consistently.
    train=arrays['origins_train'].astype(np.int64)+1
    val=arrays['origins_val'].astype(np.int64)+1
    test=arrays['origins_test'].astype(np.int64)+1
    assert [len(train),len(val),len(test)]==[21010,7003,7004]
    assert train[0]==LENGTH and train[-1]+1==val[0] and val[-1]+1==test[0]
    assert test[-1]+HORIZON==len(data)
    # No future test samples are accessible to the trainer or evaluator.
    end=int(val[-1]+HORIZON)
    assert end==int(test[0]+HORIZON-1)
    # Official adjacent windows overlap targets at the split boundary. This is
    # disclosed in protocol.json; we retain it, and never score test origins.
    arrays.update(data=data if include_test else data[:end],
                  raw=raw if include_test else raw[:end],
                  origins_train=train,origins_val=val)
    if include_test:
        release=root/'audit/test_release.json'
        if not release.is_file():raise RuntimeError('test is locked until every validation selection is frozen')
        arrays['origins_test']=test
    else:
        arrays.pop('origins_test')
    protocol=json.loads((root/'data/processed/protocol.json').read_text())
    return arrays,protocol,path


class Windows(Dataset):
    def __init__(self, arrays, origins):
        self.data=arrays['data'];self.raw=arrays['raw'];self.origins=np.asarray(origins)
    def __len__(self):return len(self.origins)
    def __getitem__(self,i):
        t=int(self.origins[i])
        return (torch.from_numpy(self.data[t-LENGTH:t]),
                torch.from_numpy(self.raw[t:t+HORIZON]),torch.tensor(t))


def loader(arrays, origins, batch_size=EVAL_BATCH, shuffle=False):
    return DataLoader(Windows(arrays,origins),batch_size=batch_size,shuffle=shuffle,
                      num_workers=0,pin_memory=torch.cuda.is_available(),drop_last=True)


@torch.no_grad()
def evaluate(base_or_kv, arrays, origins, device, bank=None, weights=None, batch_size=EVAL_BATCH):
    base_or_kv.eval()
    is_kv=bank is not None
    names=('base','kv') if is_kv else ('base',)
    scores={k:Metrics(HORIZON,arrays['mean'],arrays['std']) for k in names}
    for x,y,t in loader(arrays,origins,batch_size):
        x=x.to(device);y=y.to(device);t=t.to(device)
        if is_kv:
            pred,info=base_or_kv(x,t,bank,residual_weight=weights)
            scores['base'].add(info['base'],y);scores['kv'].add(pred,y)
        else:scores['base'].add(base_or_kv(x,t%PERIOD),y)
    return {k:v.result() for k,v in scores.items()}


def save_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def train_baseline(root,arrays,protocol,device,seed,smoke=False):
    output=root/f'runs/seed{seed}/baseline.pt'
    if output.exists():raise FileExistsError(output)
    model=TQNet(config()).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=float('inf');best_epoch=bad=0
    train_o=arrays['origins_train'];val_o=arrays['origins_val']
    if smoke:train_o=train_o[:128];val_o=val_o[:64]
    for epoch in range(1,(2 if smoke else 100)+1):
        tick=time.monotonic();last=tick;model.train();total=steps=0
        for x,y,t in loader(arrays,train_o,64,True):
            x=x.to(device);y=y.to(device);t=t.to(device)
            opt.zero_grad(set_to_none=True);pred=model(x,t%PERIOD)
            loss=masked_mae_loss(pred,y,float(arrays['mean']),float(arrays['std']))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite baseline loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            total+=float(loss.detach());steps+=1
            if time.monotonic()-last>=30:
                stage('baseline_progress',epoch=epoch,batch=steps,total=len(train_o)//64);last=time.monotonic()
        scores=evaluate(model,arrays,val_o,device);value=scores['base']['raw_masked']['mae']
        print(f'BASE_EPOCH epoch={epoch} train_mae={total/steps:.6f} val_mae={value:.6f} val_rmse={scores["base"]["raw_masked"]["rmse"]:.6f} seconds={time.monotonic()-tick:.1f}',flush=True)
        if value<best:
            best=value;best_epoch=epoch;bad=0
            save_checkpoint(output,dict(model=model.state_dict(),config=vars(config()),
                protocol=protocol,seed=seed,best_epoch=epoch,selection='full_validation_raw_masked_mae',
                optimizer='Adam',lr=.001,max_epochs=100,patience=30,loss='raw_masked_mae',
                validation=scores,version=VERSION,smoke=smoke))
        else:bad+=1
        if bad>=30:print('BASE_EARLY_STOP',flush=True);break
    payload=torch.load(output,map_location=device,weights_only=False)
    model.load_state_dict(payload['model'],strict=True)
    scores=evaluate(model,arrays,val_o,device)
    result=dict(model='TQNet',split='validation',dataset='LargeST-SD',seed=seed,
                seq_len=LENGTH,horizon=HORIZON,protocol=protocol,best_epoch=best_epoch,
                metrics=scores['base'],test_evaluated=False,test_labels_used_for_selection=False,
                checkpoint=str(output),checkpoint_sha256=sha(output),smoke=smoke,
                setup='TQNet architecture adapted to LargeST L12/H12; no published TQNet LargeST preset')
    write_json(root/f'results/baseline_validation_seed{seed}.json',result)
    print('LARGEST_TQNET_VALIDATION_RESULT '+json.dumps(result),flush=True)
    return model


def create_bank(arrays,graph,daily,weekly,device,smoke=False):
    train=arrays['origins_train']
    # Full-bank normalizers and periodic templates end strictly before the first
    # validation input. Retrieval also excludes target/history overlap per query.
    cutoff=int(arrays['origins_val'][0])-LENGTH-1
    eligible=train[train+HORIZON<=cutoff]
    chosen=select_memory_origins(eligible,12,PERIOD,horizon=HORIZON,train_end=cutoff,seq_len=LENGTH)
    if smoke:chosen=chosen[:96]
    keys=[];values=[];pool=graph['edge_pool'].to(device);last=time.monotonic()
    stage('bank_build',episodes=len(chosen),edges=len(graph['edges']),cutoff=cutoff)
    for i in range(0,len(chosen),32):
        ts_np=chosen[i:i+32];ts=torch.as_tensor(ts_np,device=device)
        x=torch.stack([torch.from_numpy(arrays['data'][t-LENGTH:t]) for t in ts_np]).to(device)
        y=torch.stack([torch.from_numpy(arrays['data'][t:t+HORIZON]) for t in ts_np]).to(device)
        with torch.no_grad():
            rh=residual_history(x,ts,daily,weekly);rv=residual_future(y,ts,daily,weekly)
            keys.append(edge_descriptors(rh,ts,pool,period=PERIOD).cpu())
            values.append(torch.einsum('bhn,ne->beh',rv,pool).cpu())
        if time.monotonic()-last>=30:stage('bank_progress',episodes=i+len(ts_np),total=len(chosen));last=time.monotonic()
    bank=build_phase_balanced_bank(torch.as_tensor(chosen),torch.cat(keys),torch.cat(values),device,8,.5)
    metadata=sampling_metadata(eligible,chosen,PERIOD,12,horizon=HORIZON,train_end=cutoff,seq_len=LENGTH) if not smoke else dict(smoke=True,episodes=len(chosen))
    return bank,metadata


@torch.no_grad()
def calibrate(model,arrays,origins,bank,device):
    # Selection uses the identical, fixed validation windows as both experts.
    grid=np.linspace(0,1,9);totals=np.zeros((HORIZON,len(grid)),np.float64)
    model.eval()
    for x,y,t in loader(arrays,origins):
        x=x.to(device);y=y.to(device);t=t.to(device)
        _,info=model(x,t,bank);mask=y.ne(0)
        for j,weight in enumerate(grid):
            raw=(info['base']+float(weight)*info['raw_residual'])*float(arrays['std'])+float(arrays['mean'])
            totals[:,j]+=(raw-y).abs().mul(mask).sum((0,2)).double().cpu().numpy()
    selected=totals.argmin(1)
    return dict(grid=grid.tolist(),weights=grid[selected].tolist(),criterion='validation raw masked MAE per forecast step',
                selection_split='validation',test_labels_used_for_selection=False)


def train_kv(root,arrays,protocol,device,seed,smoke=False):
    base_path=root/f'runs/seed{seed}/baseline.pt';base_hash=sha(base_path)
    payload=torch.load(base_path,map_location='cpu',weights_only=False)
    assert payload['protocol']==protocol and payload['seed']==seed and payload['smoke']==smoke
    base=TQNet(config());base.load_state_dict(payload['model'],strict=True);base.to(device).eval()
    graph=load_graph(root/'data/physical/sd_rn_adj.npy',root/'data/physical/sd_meta.csv',
                     expected_sensor_ids=arrays['node_ids'])
    write_json(root/'audit/hypergraph.json',dict(metadata=graph['metadata'],edges=graph['edges'],edge_metadata=graph['edge_metadata']))
    cutoff=int(arrays['origins_val'][0])-LENGTH-1
    daily,weekly=periodic_prior(arrays['data'][:cutoff],96,672)
    daily=daily.to(device);weekly=weekly.to(device)
    model=TQNetPhysicsKV(base,graph,daily,weekly,HORIZON,96).to(device)
    bank,sampling=create_bank(arrays,graph,daily,weekly,device,smoke)
    train_o=arrays['origins_train'];val_o=arrays['origins_val']
    if smoke:train_o=train_o[128:256];val_o=val_o[:64]
    frozen_hash={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    model.eval()
    x,y,t=next(iter(loader(arrays,val_o)))
    with torch.no_grad():
        pred,info=model(x.to(device),t.to(device),bank)
        torch.testing.assert_close(pred,info['base'],rtol=0,atol=0)
    stage('kv_initialized',zero_residual_equals_baseline=True,base_frozen=True,
          physical_edges=len(graph['edges']),sampling=sampling)
    opt=torch.optim.AdamW(trainable_parameters(model),lr=.002,weight_decay=.0001)
    best=float('inf');best_state=None;best_epoch=bad=0
    for epoch in range(1,(2 if smoke else 30)+1):
        tick=time.monotonic();last=tick;model.train();model.base.eval();total=steps=0
        for x,y,t in loader(arrays,train_o,32,True):
            x=x.to(device);y=y.to(device);t=t.to(device);opt.zero_grad(set_to_none=True)
            pred,_=model(x,t,bank)
            loss=masked_mae_loss(pred,y,float(arrays['mean']),float(arrays['std']))
            loss=loss+1e-4*torch.tanh(model.residual_scale).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite adapter loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(trainable_parameters(model),5);opt.step()
            total+=float(loss.detach());steps+=1
            if time.monotonic()-last>=30:
                stage('kv_progress',epoch=epoch,batch=steps,total=len(train_o)//32);last=time.monotonic()
        scores=evaluate(model,arrays,val_o,device,bank);value=scores['kv']['raw_masked']['mae']
        print(f'KV_EPOCH epoch={epoch} train_mae={total/steps:.6f} val_mae={value:.6f} base_val_mae={scores["base"]["raw_masked"]["mae"]:.6f} seconds={time.monotonic()-tick:.1f}',flush=True)
        if value<best:
            best=value;best_epoch=epoch;bad=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('base.')}
        else:bad+=1
        if bad>=5:print('KV_EARLY_STOP',flush=True);break
    if best_state is None:raise RuntimeError('no adapter selected')
    state=model.state_dict();state.update(best_state);model.load_state_dict(state,strict=True)
    uncalibrated=evaluate(model,arrays,val_o,device,bank)
    calibration=calibrate(model,arrays,val_o,bank,device)
    scores=evaluate(model,arrays,val_o,device,bank,calibration['weights'])
    with torch.no_grad():
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v.cpu(),frozen_hash[k],rtol=0,atol=0)
    assert all(p.grad is None for p in model.base.parameters()) and sha(base_path)==base_hash
    old=json.loads((root/f'results/baseline_validation_seed{seed}.json').read_text())['metrics']
    for key in ('mae','mse','rmse'):assert abs(scores['base']['raw_masked'][key]-old['raw_masked'][key])<1e-7
    result=dict(model='TQNet+PhysicsResidualKV',dataset='LargeST-SD',seed=seed,seq_len=LENGTH,horizon=HORIZON,
        split='validation',version=VERSION,protocol=protocol,metrics=scores,uncalibrated=uncalibrated,
        horizon_gate=calibration,best_epoch=best_epoch,sampling=sampling,normalization=bank.normalization_metadata,
        physical_hypergraph=graph['metadata'],base_checkpoint=str(base_path),base_checkpoint_sha256=base_hash,
        paired_same_checkpoint=True,base_weights_unchanged=True,test_evaluated=False,
        test_labels_used_for_selection=False,memory_train_only=True,memory_template_fit_end=cutoff,
        validation_scope='full official validation; used for checkpoint and gate selection; not independent test evidence',
        smoke=smoke,physical_gpu=2,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    for key in ('mae','mse','rmse'):
        base_metric=scores['base']['raw_masked'][key];kv_metric=scores['kv']['raw_masked'][key]
        result.setdefault('relative_gain',{})[key]=(base_metric-kv_metric)/base_metric
    result['screen_decision']='FROZEN_RECIPE_THREE_SEED_CONFIRMATION'
    save_checkpoint(root/f'runs/seed{seed}/adapter.pt',dict(adapter=best_state,protocol=protocol,
        base_checkpoint_sha256=base_hash,seed=seed,best_epoch=best_epoch,horizon_gate=calibration,
        sampling=sampling,normalization=bank.normalization_metadata,version=VERSION,smoke=smoke))
    write_json(root/f'results/paired_validation_seed{seed}.json',result)
    print('LARGEST_PAIRED_VALIDATION_RESULT '+json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--stage',choices=['baseline','kv'],required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,default=2024);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise RuntimeError('only physical GPU2 is authorized')
    if not torch.cuda.is_available():raise RuntimeError('CUDA2 unavailable')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    root=Path(a.root).resolve()
    if root!=Path(__file__).resolve().parent:raise RuntimeError('root must be isolated project directory')
    if (root/'audit/test_release.json').exists():raise RuntimeError('validation selections are frozen; training is locked')
    if (root/f'results/paired_validation_seed{a.seed}.json').exists():raise FileExistsError('completed pair; do not overwrite')
    arrays,protocol,path=prepared(root)
    stage('protocol',version=VERSION,physical_gpu=2,seed=a.seed,protocol=protocol,data_sha256=sha(path),
          train_windows=len(arrays['origins_train']),validation_windows=len(arrays['origins_val']),
          validation_used_windows=len(arrays['origins_val'])//64*64,test_evaluated=False,
          model_configuration=vars(config()),training_objective='raw_masked_mae')
    device=torch.device('cuda:0')
    if a.stage=='baseline':train_baseline(root,arrays,protocol,device,a.seed,False)
    else:train_kv(root,arrays,protocol,device,a.seed,False)


if __name__=='__main__':main()
