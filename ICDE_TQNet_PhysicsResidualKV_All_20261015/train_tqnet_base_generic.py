from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.tqnet import Model as TQNet


class Windows(Dataset):
    def __init__(self, data, origins, horizon): self.data, self.origins, self.horizon = data, np.asarray(origins), horizon
    def __len__(self): return len(self.origins)
    def __getitem__(self, idx):
        t=int(self.origins[idx]); return torch.from_numpy(self.data[t-96:t]), torch.from_numpy(self.data[t:t+self.horizon]), torch.tensor(t)


def cfg(nodes, horizon, cycle): return SimpleNamespace(seq_len=96,pred_len=horizon,enc_in=nodes,cycle=cycle,model_type='mlp',d_model=512,dropout=0.0,use_revin=1)


@torch.no_grad()
def evalm(model, loader, device):
    model.eval(); se=ae=n=0
    for x,y,t in loader:
        x=x.to(device).float(); y=y.to(device).float(); pred=model(x,t.to(device).long()%model.cycle_len); e=pred-y; se+=float(e.square().sum(dtype=torch.float64)); ae+=float(e.abs().sum(dtype=torch.float64)); n+=y.numel()
    return {'mse':se/max(n,1),'mae':ae/max(n,1)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data_path',required=True); p.add_argument('--dataset',required=True); p.add_argument('--num_nodes',type=int,required=True); p.add_argument('--horizon',type=int,required=True); p.add_argument('--cycle',type=int,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--split',choices=['60/20/20','70/10/20'],required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    with np.load(a.data_path,allow_pickle=False) as f: data=f['data'][:,:,0].astype(np.float32)
    train_end=int(len(data)*(.6 if a.split=='60/20/20' else .7)); val_end=train_end+int(len(data)*(.2 if a.split=='60/20/20' else .1)); model=TQNet(cfg(a.num_nodes,a.horizon,a.cycle)).to(device).float()
    train=DataLoader(Windows(data,np.arange(96,train_end-a.horizon+1),a.horizon),a.batch_size,True,num_workers=2,pin_memory=True); val=DataLoader(Windows(data,np.arange(train_end,val_end-a.horizon+1),a.horizon),a.batch_size,False,num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=1e99; best_state=None; bad=0
    for ep in range(1,a.epochs+1):
        st=time.monotonic(); model.train(); total=steps=0
        for x,y,t in train:
            x=x.to(device).float(); y=y.to(device).float(); opt.zero_grad(set_to_none=True); pred=model(x,t.to(device).long()%model.cycle_len); loss=(pred-y).square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total+=float(loss.detach()); steps+=1
        vm=evalm(model,val,device); print(f'epoch={ep} train_mse={total/max(steps,1):.6f} val_mse={vm["mse"]:.6f} val_mae={vm["mae"]:.6f} seconds={time.monotonic()-st:.1f}',flush=True)
        if vm['mse']<best: best=vm['mse']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=5: print('early_stop',flush=True); break
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':best_state,'dataset':a.dataset,'horizon':a.horizon,'cycle':a.cycle,'val_mse':best},out); print(json.dumps({'output':str(out),'dataset':a.dataset,'horizon':a.horizon,'val_mse':best}),flush=True)

if __name__=='__main__': main()
