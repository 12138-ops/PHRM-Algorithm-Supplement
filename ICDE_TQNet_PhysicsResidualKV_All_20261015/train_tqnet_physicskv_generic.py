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

from generic_graph import load_graph
from models.tqnet import Model as TQNet
from physics_kv import (
    PhysicalKVBank,
    TQNetPhysicsKV,
    edge_descriptors,
    periodic_prior,
    residual_future,
    residual_history,
    trainable_parameters,
)


GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, 9))


class Windows(Dataset):
    def __init__(self, data, origins, horizon):
        self.data, self.origins, self.horizon = data, np.asarray(origins), int(horizon)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        t = int(self.origins[idx])
        return torch.from_numpy(self.data[t - 96:t]), torch.from_numpy(self.data[t:t + self.horizon]), torch.tensor(t)


def loader(data, origins, horizon, batch_size, workers, shuffle):
    return DataLoader(Windows(data, origins, horizon), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def config(nodes, horizon, cycle):
    return SimpleNamespace(seq_len=96, pred_len=horizon, enc_in=nodes, cycle=cycle,
                           model_type="mlp", d_model=512, dropout=0.0, use_revin=1)


def load_base(checkpoint, nodes, horizon, cycle, device):
    model = TQNet(config(nodes, horizon, cycle)).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"TQNet checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def build_bank(data, origins, horizon, graph, daily, weekly, batch_size, device):
    chunks, values = [], []
    pool = graph["edge_pool"].to(device)
    for start in range(0, len(origins), batch_size):
        ts_np = origins[start:start + batch_size]
        ts = torch.as_tensor(ts_np, dtype=torch.long, device=device)
        x = torch.stack([torch.from_numpy(data[int(t) - 96:int(t)]) for t in ts_np]).to(device)
        y = torch.stack([torch.from_numpy(data[int(t):int(t) + horizon]) for t in ts_np]).to(device)
        rh = residual_history(x, ts, daily, weekly)
        rv = residual_future(y, ts, daily, weekly)
        chunks.append(edge_descriptors(rh, ts, pool).cpu())
        values.append(torch.einsum("bhn,ne->beh", rv, pool).cpu())
    keys = torch.cat(chunks, 0) if chunks else torch.empty(0, len(graph["edges"]), 13)
    vals = torch.cat(values, 0) if values else torch.empty(0, len(graph["edges"]), horizon)
    return PhysicalKVBank(torch.as_tensor(origins), keys, vals, device, 8, 0.5)


@torch.no_grad()
def evaluate(model, data_loader, bank, device, residual_weight=None):
    model.eval(); model.base.eval(); totals={"trained":[0.,0.],"base":[0.,0.]}; count=0
    for x, y, origins in data_loader:
        x=x.to(device).float(); y=y.to(device).float(); origins=origins.to(device).long()
        pred, info=model(x, origins, bank, residual_weight=residual_weight)
        for name, value in (("trained", pred), ("base", info["base"])):
            err=value-y; totals[name][0]+=float(err.square().sum(dtype=torch.float64)); totals[name][1]+=float(err.abs().sum(dtype=torch.float64))
        count += y.numel()
    return {k:{"mse":a/max(count,1),"mae":b/max(count,1)} for k,(a,b) in totals.items()}


@torch.no_grad()
def calibrate(model, data_loader, bank, device):
    model.eval(); h=model.horizon; mse=torch.zeros(h,len(GRID),dtype=torch.float64); mae=torch.zeros_like(mse); count=0
    for x,y,origins in data_loader:
        x=x.to(device).float(); y=y.to(device).float(); origins=origins.to(device).long(); _,info=model(x,origins,bank)
        for j,w in enumerate(GRID):
            err=info["base"]+w*info["raw_residual"]-y; mse[:,j]+=err.square().sum((0,2)).cpu(); mae[:,j]+=err.abs().sum((0,2)).cpu()
        count += int(y.shape[0]*y.shape[2])
    mse/=max(count,1); mae/=max(count,1); best=torch.argmin(mse,1); weights=[GRID[int(i)] for i in best.tolist()]
    return {"grid":list(GRID),"weights":weights,"best_index":best.tolist(),"validation_mse":float(mse[torch.arange(h),best].mean()),"validation_mae":float(mae[torch.arange(h),best].mean()),"selection_split":"validation","test_labels_used_for_selection":False}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="."); p.add_argument("--dataset",required=True); p.add_argument("--data_path",required=True); p.add_argument("--edge_csv",required=True); p.add_argument("--base_checkpoint",required=True); p.add_argument("--num_nodes",type=int,required=True); p.add_argument("--horizon",type=int,required=True); p.add_argument("--cycle",type=int,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--split",choices=["60/20/20","70/10/20"],required=True); p.add_argument("--epochs",type=int,default=20); p.add_argument("--patience",type=int,default=5); p.add_argument("--batch_size",type=int,default=4); p.add_argument("--memory_stride",type=int,default=12); p.add_argument("--device",default="cuda:0"); p.add_argument("--runs_root",default="runs_tqnet")
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); root=Path(a.root).resolve(); device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    with np.load(a.data_path,allow_pickle=False) as f: data=f["data"][:,:,0].astype(np.float32)
    ratio=.6 if a.split=="60/20/20" else .7; train_end=int(len(data)*ratio); val_end=train_end+int(len(data)*(.2 if a.split=="60/20/20" else .1)); cycle=a.cycle
    mean=data[:train_end].mean(0); std=np.maximum(data[:train_end].std(0),1e-3); daily,weekly=periodic_prior(data[:train_end],daily_period=cycle,weekly_period=cycle*7); daily=daily.to(device); weekly=weekly.to(device)
    graph=load_graph(Path(a.edge_csv),a.num_nodes); base=load_base(a.base_checkpoint,a.num_nodes,a.horizon,cycle,device); model=TQNetPhysicsKV(base,graph,daily,weekly,a.horizon,96).to(device)
    train_o=np.arange(96,train_end-a.horizon+1); val_o=np.arange(train_end,val_end-a.horizon+1); test_o=np.arange(val_end,len(data)-a.horizon+1); bank_o=train_o[::max(1,a.memory_stride)]
    bank=build_bank(data,bank_o,a.horizon,graph,daily,weekly,a.batch_size,device); train=loader(data,train_o,a.horizon,a.batch_size,2,True); val=loader(data,val_o,a.horizon,a.batch_size,2,False); test=loader(data,test_o,a.horizon,a.batch_size,2,False)
    opt=torch.optim.AdamW(trainable_parameters(model),lr=2e-3,weight_decay=1e-4); best=float("inf"); best_state=None; best_epoch=0; bad=0
    for epoch in range(1,a.epochs+1):
        st=time.monotonic(); model.train(); model.base.eval(); total=0.; steps=0
        for x,y,o in train:
            x=x.to(device).float(); y=y.to(device).float(); o=o.to(device).long(); opt.zero_grad(set_to_none=True); pred,info=model(x,o,bank); loss=(pred-y).square().mean()+1e-4*torch.tanh(model.residual_scale).square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable_parameters(model),5); opt.step(); total+=float(loss.detach()); steps+=1
        vm=evaluate(model,val,bank,device); print(f"epoch={epoch} train={total/max(steps,1):.6f} val_mse={vm['trained']['mse']:.6f} seconds={time.monotonic()-st:.1f}",flush=True)
        if vm["trained"]["mse"]<best: best=vm["trained"]["mse"]; best_epoch=epoch; bad=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('base.')}
        else: bad+=1
        if bad>=a.patience: print('early_stop',flush=True); break
    cur=model.state_dict(); cur.update(best_state); model.load_state_dict(cur,strict=False); calibration=calibrate(model,val,bank,device); val_metrics=evaluate(model,val,bank,device,torch.tensor(calibration['weights'],device=device)); test_metrics=evaluate(model,test,bank,device,torch.tensor(calibration['weights'],device=device))
    out=root/a.runs_root/a.dataset/f"h{a.horizon}"/f"seed{a.seed}"; out.mkdir(parents=True,exist_ok=True); torch.save({"adapter":best_state,"base_checkpoint":str(a.base_checkpoint),"best_epoch":best_epoch},out/'adapter.pt'); result={"model":"TQNetPhysicsResidualKV","dataset":a.dataset,"horizon":a.horizon,"split":"test","protocol":a.split,"seed":a.seed,"train_end":train_end,"val_end":val_end,"memory_train_only":True,"test_labels_used_for_selection":False,"physical_edges":len(graph['edges']),"horizon_gate":calibration,"validation":val_metrics,"test":test_metrics,"base_checkpoint":str(a.base_checkpoint)}; (out/'result.json').write_text(json.dumps(result,indent=2)); print('TQNET_PHYSICS_KV_RESULT '+json.dumps(result),flush=True)

if __name__=='__main__': main()
