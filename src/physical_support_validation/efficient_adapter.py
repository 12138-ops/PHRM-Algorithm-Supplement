"""Algebraically unchanged adapter with bounded reconstruction intermediates."""
import torch
from physics_kv import StrongBackbonePhysicsKV,residual_history,edge_descriptors,_phase_value

def reconstruct(incidence,edge_memory,edge_weight):
 # Weight the small BxExH tensor before contracting nodes and edges. The
 # original three-operand einsum may materialize BxNxExH without opt_einsum.
 return torch.einsum('ne,beh->bhn',incidence,edge_memory*edge_weight[:,:,None])

class EfficientStrongKV(StrongBackbonePhysicsKV):
 def forward(self,x,origins,bank,disable_memory=False,residual_weight=None):
  with torch.no_grad():base=self.predict_base(x,origins)
  rh=residual_history(x,origins,self.daily,self.weekly)
  query=edge_descriptors(rh,origins,self.edge_pool,period=self.daily.shape[0])
  memory,var,support,nearest=bank.retrieve(query,origins,x.shape[1])
  edge_weight=support/(1+nearest.squeeze(-1));edge_weight=edge_weight/edge_weight.sum(-1,keepdim=True).clamp_min(1e-6)
  denom=torch.einsum('ne,be->bn',self.incidence,edge_weight).clamp_min(1e-6)
  node_memory=reconstruct(self.incidence,memory,edge_weight)/denom[:,None,:]
  periodic=_phase_value(self.daily,self.weekly,origins,torch.arange(self.horizon,device=x.device))
  physical=periodic+node_memory;diff=physical-base
  features=torch.stack([base.mean(1),base.std(1,unbiased=False),diff.mean(1),diff.std(1,unbiased=False),x[:,-1],node_memory.mean(1)],dim=-1)
  gate=self.gate(features).sigmoid().transpose(1,2);scale=self.residual_scale.tanh()[None,:,None]
  if disable_memory:gate=gate*0
  raw=scale*gate*diff
  if residual_weight is None:residual=raw
  else:
   w=torch.as_tensor(residual_weight,dtype=raw.dtype,device=raw.device)
   if w.ndim==0:w=w.view(1,1,1)
   elif w.ndim==1:
    if w.numel()!=self.horizon:raise ValueError('Invalid calibration horizon')
    w=w.view(1,self.horizon,1)
   elif w.ndim==2:
    if w.shape!=(raw.shape[0],self.horizon):raise ValueError('Invalid per-sample calibration')
    w=w.unsqueeze(-1)
   else:raise ValueError('Invalid calibration dimensions')
   residual=raw*w
  return base+residual,dict(base=base,physical=physical,diff=diff,gate=gate,support=support,edge_var=var,nearest=nearest,raw_residual=raw,residual=residual)
