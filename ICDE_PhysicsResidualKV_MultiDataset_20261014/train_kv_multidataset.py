"""Dataset-parameterized PhysicsResidualKV validation runner."""
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from backbone import physical_candidates,graph_tensors,periodic_prior
from residual_kv import PhysicsResidualKV,ResidualKVBank,local_keys,set_scope

class W(Dataset):
 def __init__(self,raw,o,h): self.raw,self.o,self.h=raw,np.asarray(o),h
 def __len__(self): return len(self.o)
 def __getitem__(self,i): t=int(self.o[i]); return self.raw[t-96:t],self.raw[t:t+self.h],t
def dl(raw,o,b,w,shuffle,h): return DataLoader(W(raw,o,h),b,shuffle,num_workers=w,persistent_workers=w>0,pin_memory=True)
def batch(b,d): x,y,t=b; return x.to(d).float(),y.to(d).float(),t.to(d).long()
def load_model(root,raw,end,graph,a,dev,h,ds):
 mean=raw[:end].mean(0).astype(np.float32); std=np.maximum(raw[:end].std(0),1e-3).astype(np.float32); daily,weekly=periodic_prior(raw[:end],mean,std)
 m=PhysicsResidualKV(graph,mean,std,daily.to(dev),weekly.to(dev),96,h,a.hidden,'physical',a.topk,a.temperature).to(dev)
 ref=root/'references'/ds/f'h{h}_temporal.pt'; ck=torch.load(ref,map_location='cpu',weights_only=False); state=ck['model'] if 'model' in ck else ck
 need={k for k in m.state_dict() if k in state and (k in ('daily','weekly','norm_mean','norm_std','daily_reference','weekly_reference') or k.startswith(('temporal_encoder.','temporal_head.')))}; m.load_state_dict({k:state[k] for k in need},strict=False)
 return m,torch.tensor(mean,device=dev),torch.tensor(std,device=dev),daily.to(dev),weekly.to(dev)
@torch.no_grad()
def make_bank(raw,end,graph,a,dev,h,mean,std,daily,weekly):
 inc,fam,anc=graph_tensors(graph); pool=inc/inc.sum(0,keepdim=True).clamp_min(1); origins=np.arange(96,end-h+1,a.memory_stride); out=[]
 for b in dl(raw,origins,a.batch_size,0,False,h):
  x,y,t=batch(b,dev); hi=t[:,None]+torch.arange(-96,0,device=dev)[None,:]; fi=t[:,None]+torch.arange(h,device=dev)[None,:]; rh=(x-mean-std*(daily[hi%288]+weekly[hi%2016]))/std; rv=(y-mean-std*(daily[fi%288]+weekly[fi%2016]))/std; out.append((t.cpu(),local_keys(rh,t,pool.to(dev)).cpu(),rv.cpu()))
 return ResidualKVBank({'origins':torch.cat([z[0] for z in out]),'keys':torch.cat([z[1] for z in out]),'values':torch.cat([z[2] for z in out]),'incidence':inc,'family':fam,'anchors':anc},dev)
@torch.no_grad()
def evalm(m,loader,mean,std,dev,bank):
 m.eval(); s={'trained':[0.,0.],'base':[0.,0.]}; cnt=0
 for b in loader:
  x,y,t=batch(b,dev); p,c=m(x,t,bank)
  for k,v in {'trained':p,'base':c['base']}.items(): e=(v-y)/std; s[k][0]+=float(e.square().sum(dtype=torch.float64)); s[k][1]+=float(e.abs().sum(dtype=torch.float64))
  cnt+=y.numel()
 return {k:{'mse':a/cnt,'mae':b/cnt} for k,(a,b) in s.items()}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--dataset',required=True); p.add_argument('--data_path'); p.add_argument('--edge_csv'); p.add_argument('--num_nodes',type=int); p.add_argument('--horizon',type=int,required=True); p.add_argument('--seed',type=int,default=2024); p.add_argument('--epochs',type=int,default=25); p.add_argument('--warmup',type=int,default=5); p.add_argument('--patience',type=int,default=5); p.add_argument('--batch_size',type=int,default=16); p.add_argument('--workers',type=int,default=2); p.add_argument('--memory_stride',type=int,default=12); p.add_argument('--topk',type=int,default=8); p.add_argument('--temperature',type=float,default=.5); p.add_argument('--hidden',type=int,default=128); p.add_argument('--lr',type=float,default=.001); p.add_argument('--edge_budget',type=int,default=512); p.add_argument('--max_edge_size',type=int,default=16); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); root=Path(a.root).resolve(); ds=a.dataset.upper(); h=a.horizon; dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 data=Path(a.data_path) if a.data_path else root/'dataset/raw'/f'{ds}.npz'; edge=Path(a.edge_csv) if a.edge_csv else root/'dataset/physical'/f'{ds}.csv'
 with np.load(data,allow_pickle=False) as f: raw=f['data'][:,:,0].astype(np.float32)
 n=int(a.num_nodes or raw.shape[1]); end=int(len(raw)*.6); vend=end+int(len(raw)*.2); graph=physical_candidates(edge,n,a.edge_budget,a.max_edge_size); m,mean,std,daily,weekly=load_model(root,raw,end,graph,a,dev,h,ds); bank=make_bank(raw,end,graph,a,dev,h,mean,std,daily,weekly); train=dl(raw,np.arange(96,end-h+1),a.batch_size,a.workers,True,h); val=dl(raw,np.arange(end,vend-h+1),a.batch_size,a.workers,False,h); best=1e99; state=None; be=0; bad=0
 ep_counter=0
 for stage,epochs in (('warmup',a.warmup),('joint',a.epochs)):
  set_scope(m,stage=='warmup'); opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=a.lr,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(epochs,1))
  for _ in range(epochs):
   m.train(); total=steps=0.; st=time.monotonic()
   for b in train:
    x,y,t=batch(b,dev); opt.zero_grad(set_to_none=True); pred,c=m(x,t,bank,disable_memory=(random.random()<.1)); loss=(((pred-y)/std)**2).mean()+1e-4*c['delta'].square().mean(); loss.backward(); torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],5); opt.step(); total+=float(loss.detach()); steps+=1
   sch.step(); vm=evalm(m,val,mean,std,dev,bank)['trained']; ep_counter+=1; ep=ep_counter; print(f'epoch={ep} train_mse={total/max(steps,1):.6f} val_mse={vm["mse"]:.6f} val_mae={vm["mae"]:.6f} seconds={time.monotonic()-st:.1f}',flush=True)
   if vm['mse']<best: best=vm['mse']; be=ep; bad=0; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
   else: bad+=1
   if stage=='joint' and bad>=a.patience: print('early_stop',flush=True); break
  if stage=='joint' and bad>=a.patience: break
 m.load_state_dict(state); metrics=evalm(m,val,mean,std,dev,bank); out=root/'runs'/ds/f'h{h}'/f'seed{a.seed}'; out.mkdir(parents=True,exist_ok=True); torch.save({'model':state,'best_epoch':be,'metrics':metrics},out/'physical.pt'); r={'model':'PhysicsResidualKV','dataset':ds,'horizon':h,'split':'validation','seed':a.seed,'windows':len(np.arange(end,vend-h+1)),'memory_episodes':int(bank.origins.numel()),'metrics':metrics,'best_epoch':be}; (out/'validation_result.json').write_text(json.dumps(r,indent=2)); print('PHYSICS_RESIDUAL_KV_RESULT '+json.dumps(r),flush=True)
if __name__=='__main__': main()
