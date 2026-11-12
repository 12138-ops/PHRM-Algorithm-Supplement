from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.tqnet import Model as TQNet
from protocol import SPECS, load_data


class Windows(Dataset):
    def __init__(self, data, origins, horizon): self.data, self.origins, self.horizon = data, np.asarray(origins), horizon
    def __len__(self): return len(self.origins)
    def __getitem__(self, idx):
        t=int(self.origins[idx]); return torch.from_numpy(self.data[t-96:t]), torch.from_numpy(self.data[t:t+self.horizon]), torch.tensor(t)


def cfg(nodes, horizon, cycle, use_revin=1, dropout=0): return SimpleNamespace(seq_len=96,pred_len=horizon,enc_in=nodes,cycle=cycle,model_type='mlp',d_model=512,dropout=dropout,use_revin=use_revin)


@torch.no_grad()
def evalm(model, loader, device):
    model.eval(); se=ae=n=0
    for x,y,t in loader:
        x=x.to(device).float(); y=y.to(device).float(); pred=model(x,t.to(device).long()%model.cycle_len); e=pred-y; se+=float(e.square().sum(dtype=torch.float64)); ae+=float(e.abs().sum(dtype=torch.float64)); n+=y.numel()
    return {'mse':se/max(n,1),'mae':ae/max(n,1)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_path',required=True); p.add_argument('--dataset',required=True); p.add_argument('--num_nodes',type=int,required=True); p.add_argument('--horizon',type=int,required=True); p.add_argument('--cycle',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--split',choices=['60/20/20','70/10/20','ETT-12/4/4-months'],required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; device=torch.device(a.device)
    spec=SPECS[a.dataset]
    if (a.num_nodes,a.cycle,a.split)!=(spec['nodes'],spec['cycle'],spec['split']): raise ValueError('protocol configuration mismatch')
    prepared=load_data(a.dataset,a.data_path); data=prepared.data; prepared.save_scaler(Path(a.output).parents[3])
    train_o,val_o,_=prepared.origins(a.horizon); model=TQNet(cfg(a.num_nodes,a.horizon,a.cycle,spec['use_revin'],spec.get('dropout',0))).to(device).float()
    print('PROTOCOL '+json.dumps(prepared.metadata),flush=True)
    train=DataLoader(Windows(data,train_o,a.horizon),a.batch_size,True,num_workers=2,pin_memory=True,drop_last=True); val=DataLoader(Windows(data,val_o,a.horizon),a.batch_size,False,num_workers=2,pin_memory=True)
    base_lr=spec.get('base_lr',.003)
    opt=torch.optim.Adam(model.parameters(),lr=base_lr); best=1e99; best_state=None; bad=0
    for ep in range(1,a.epochs+1):
        st=time.monotonic(); model.train(); total=steps=0
        for x,y,t in train:
            x=x.to(device).float(); y=y.to(device).float(); opt.zero_grad(set_to_none=True); pred=model(x,t.to(device).long()%model.cycle_len); loss=(pred-y).square().mean(); loss.backward(); opt.step(); total+=float(loss.detach()); steps+=1
        vm=evalm(model,val,device); print(f'epoch={ep} train_mse={total/max(steps,1):.6f} val_mse={vm["mse"]:.6f} val_mae={vm["mae"]:.6f} seconds={time.monotonic()-st:.1f}',flush=True)
        if vm['mse']<best: best=vm['mse']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=5: print('early_stop',flush=True); break
        for group in opt.param_groups: group['lr']=base_lr if ep<3 else base_lr*(.8**(ep-3))
    if best_state is None or not np.isfinite(best): raise RuntimeError('no finite baseline checkpoint')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':best_state,'dataset':a.dataset,'horizon':a.horizon,'cycle':a.cycle,'val_mse':best,'protocol_fingerprint':prepared.metadata['fingerprint'],'preprocessing':prepared.metadata,'optimizer':'Adam','initial_lr':base_lr,'lradj':'type3','batch_size':a.batch_size,'max_epochs':a.epochs,'seed':a.seed},out)
    print(json.dumps({'output':str(out),'dataset':a.dataset,'horizon':a.horizon,'val_mse':best}),flush=True)

if __name__=='__main__': main()
