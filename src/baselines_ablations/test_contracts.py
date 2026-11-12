import copy,unittest
import numpy as np
import torch
from torch import nn
from protocol import Data
from ablation_model import Adapter,ClockBank,GlobalBank,rewired_graph,tensors,VARIANTS
from prepare import jobs
import frozen_physics as frozen

class Base(nn.Module):
 def __init__(self,h):super().__init__();self.h=h;self.cycle_len=288;self.layer=nn.Linear(12,h)
 def forward(self,x,cycle):return self.layer(x[:,-12:].transpose(1,2)).transpose(1,2)

class Contracts(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(4);self.n=8;self.h=4;self.g=tensors([[(i+j)%8 for j in (0,1,3)] for i in range(8)],8);self.daily=torch.rand(288,8);self.week=torch.rand(2016,8);self.x=torch.rand(3,96,8);self.o=torch.tensor([1000,1100,1200]);self.keys=torch.rand(4,8,13);self.values=torch.rand(4,8,4);self.bank=frozen.PhysicalKVBank(torch.tensor([96,200,300,400]),self.keys,self.values,'cpu')
 def make(self,v):return Adapter(Base(self.h),self.g,self.daily,self.week,self.h,v)
 def test_initial_forecast_equals_backbone_and_base_stays_eval(self):
  for v in VARIANTS:
   if v=='global_memory':continue
   m=self.make(v);m.train();self.assertFalse(m.base.training)
   pred,_=m(self.x,self.o,self.bank);torch.testing.assert_close(pred,m.base(self.x,self.o%288),atol=0,rtol=0)
 def test_periodic_only_is_not_base_when_trained(self):
  m=self.make('periodic_only');m.residual_scale.data.fill_(.5);a,_=m(self.x,self.o,None)
  self.assertGreater(float((a-m.base(self.x,self.o%288)).abs().max()),.001)
 def test_cached_main_outputs_and_gradients_match_frozen_forward(self):
  m=self.make('full');m.residual_scale.data.fill_(.2);r=copy.deepcopy(m)
  base,diff,features=m.decompose(self.x,self.o,self.bank);a,_=m.mix(base,diff,features)
  b,_=frozen.TQNetPhysicsKV.forward(r,self.x,self.o,self.bank)
  torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6);a.square().mean().backward();b.square().mean().backward()
  for (n,p),(rn,rp) in zip(m.named_parameters(),r.named_parameters()):
   if p.requires_grad:torch.testing.assert_close(p.grad,rp.grad,rtol=2e-5,atol=1e-6)
 def test_parameter_matched_mlp_has_no_memory_dependency(self):
  m=self.make('residual_mlp');r=self.make('full');self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),sum(p.numel() for p in r.parameters() if p.requires_grad))
  m.residual_scale.data.fill_(.2);a,_=m(self.x,self.o,None);b,_=m(self.x,self.o,object());torch.testing.assert_close(a,b)
 def test_clock_retrieval_ignores_state_coordinates(self):
  bank=ClockBank(torch.tensor([96,200,300,400]),self.keys,self.values,'cpu',2,.5);q=self.keys[:3].clone();r=q.clone();r[...,:11]+=100
  for a,b in zip(bank.retrieve(q,self.o,96),bank.retrieve(r,self.o,96)):torch.testing.assert_close(a,b)
 def test_temporal_exclusion_before_any_memory_is_available(self):
  z=self.bank.retrieve(self.keys[:1],torch.tensor([100]),96)[0];self.assertEqual(float(z.abs().sum()),0)
 def test_global_memory_retains_distinct_node_trajectories(self):
  values=torch.arange(3*4*8,dtype=torch.float32).reshape(3,4,8);keys=torch.rand(3,13);bank=GlobalBank(torch.tensor([100,200,300]),keys,values,'cpu',1,.5)
  a=bank.retrieve_global(keys[:1],torch.tensor([1000]),96);torch.testing.assert_close(a[0],values[0]);self.assertNotEqual(float(a[0,0,0]),float(a[0,0,1]))
 def test_rewiring_preserves_degrees_sizes_coverage(self):
  n=40;g=tensors([sorted({i,(i+1)%n,(i+3)%n,(i+7)%n}) for i in range(n)],n);r=rewired_graph(g)
  torch.testing.assert_close(g['incidence'].sum(0),r['incidence'].sum(0));torch.testing.assert_close(g['incidence'].sum(1),r['incidence'].sum(1));self.assertGreater(r['metadata']['changed_edges'],30)
 def test_split_endpoints_keep_targets_inside_each_partition(self):
  data=Data(np.zeros((1000,8)),np.zeros((1000,1)),dict(shape=[1000,8],train_end=600,val_end=799,test_start=800))
  self.assertEqual(data.origins(24,'train')[-1]+24,600);self.assertEqual(data.origins(24,'val')[0],600);self.assertEqual(data.origins(24,'val')[-1]+24,799);self.assertEqual(data.origins(24,'test')[-1]+24,1000)
 def test_complete_predeclared_job_graph_excludes_traffic(self):
  js=jobs();self.assertEqual(len(js),349);self.assertEqual(len({j['id'] for j in js}),349);self.assertTrue(all(j['dataset']!='Traffic' for j in js));ids={j['id'] for j in js};self.assertTrue(all(set(j['dependencies'])<=ids for j in js))

if __name__=='__main__':unittest.main()
