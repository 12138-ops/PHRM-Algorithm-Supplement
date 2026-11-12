"""Small validation-only integration check; outputs never enter formal tables."""
import gc,json,shutil,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import *
from protocol import Windows,load
from ablations import setup,create_cache,cached_loader,evaluate_cached
from baselines import build,forward,recipe_fields

def main():
 gpu,device=require_gpu();reports=[];directory=ROOT/'audit/smoke'
 for ds,h,v in [('PEMS08',24,'full'),('PEMS08',24,'global_memory'),('PEMS08',24,'periodic_only'),('PEMS07',96,'full')]:
  job=dict(id=f'smoke_{ds}_{v}',dataset=ds,horizon=h,seed=2024,variant=v)
  data,m,bank,graph,ck,oo=setup(job,device);original=sha(ck);train_o=data.origins(h,'train')[-32:];val_o=data.origins(h,'val')[:16];path=directory/job['id'];path.mkdir(parents=True,exist_ok=True)
  for split,origins in [('train',train_o),('val',val_o)]:create_cache(path/split,data,origins,h,m,bank,4,device,job['id'])
  dl=cached_loader(path/'train',data,train_o,h,4,True);vl=cached_loader(path/'val',data,val_o,h,4,False);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.002)
  for b,d,f,y in dl:
   opt.zero_grad(set_to_none=True);pred,_=m.mix(b.to(device),d.to(device),f.to(device));loss=(pred-y.to(device)).square().mean();loss.backward();opt.step()
  metrics=evaluate_cached(m,vl,device,calibrate=True);assert np.isfinite(metrics['trained']['mse']);weights=torch.tensor(metrics['calibration']['weights'],device=device);cached=evaluate_cached(m,vl,device,weights)
  state={k:value.detach().cpu().clone() for k,value in m.state_dict().items() if not k.startswith('base.')};torch.save(state,path/'adapter.pt');current=m.state_dict();current.update(torch.load(path/'adapter.pt',map_location='cpu',weights_only=True));m.load_state_dict(current,strict=True);m.eval()
  se=ae=count=0
  with torch.no_grad():
   for x,y,o,marks in DataLoader(Windows(data,val_o,h),batch_size=4):
    pred,_=m(x.to(device),o.to(device),bank,weights);e=pred-y.to(device);se+=float(e.square().sum(dtype=torch.float64));ae+=float(e.abs().sum(dtype=torch.float64));count+=e.numel()
  assert abs(se/count-cached['trained']['mse'])<1e-6 and abs(ae/count-cached['trained']['mae'])<1e-6
  assert sha(ck)==original and all(p.grad is None for p in m.base.parameters())
  r=dict(dataset=ds,variant=v,checkpoint_round_trip=True,cache_and_live_validation_agree=True,base_unchanged=True,validation_windows=len(val_o),test_evaluated=False);reports.append(r);log('SMOKE_PIPELINE',**r)
  del dl,vl,m,bank,opt;gc.collect();torch.cuda.empty_cache();shutil.rmtree(path)
 # Exercise checkpoint payload loading for a new baseline, as used after fitting.
 for ds,name in [('PEMS04','GTR'),('PEMS08','PatchTST')]:
  from baselines import recipe
  a=recipe(ds,name,96) if name=='GTR' else recipe_fields(next(r['args'] for r in old_records() if r['dataset']==ds and r['model']==name and r['horizon']==96 and r['seed']==2024))
  seed_all(2024);data=load(ds);m=build(name,a).to(device);opt=torch.optim.Adam(m.parameters(),lr=a['learning_rate']);origins=data.origins(96,'train')[-4:];x=torch.from_numpy(np.stack([data.values[t-96:t] for t in origins])).to(device);y=torch.from_numpy(np.stack([data.values[t:t+96] for t in origins])).to(device);marks=torch.from_numpy(np.stack([data.marks[t-96:t] for t in origins])).to(device);o=torch.tensor(origins,device=device)
  pred=forward(m,name,x,o,marks,a['cycle']);loss=(pred-y).square().mean();loss.backward();opt.step();p=directory/f'{ds}_{name}.pt';torch.save(dict(model=m.state_dict(),recipe=a,epoch=1),p);state=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(state['model'],strict=True);assert torch.isfinite(loss);p.unlink()
  reports.append(dict(dataset=ds,model=name,finite_backward=True,checkpoint_round_trip=True,test_evaluated=False));del pred,loss,m,opt,x,y,marks,o;gc.collect();torch.cuda.empty_cache()
 write_json(ROOT/'audit/pipeline_smoke.json',dict(passed=True,physical_gpu=gpu,checks=reports,test_evaluated=False));log('PIPELINE_SMOKE_COMPLETE',checks=len(reports))

if __name__=='__main__':main()
