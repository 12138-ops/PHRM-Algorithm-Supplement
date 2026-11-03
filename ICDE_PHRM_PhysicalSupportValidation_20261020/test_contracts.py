"""Meaningful checks of incidence, sampling, storage and split invariants."""
import unittest
import numpy as np
from support_graphs import local_chain,margins
from memory_budget import select_count,episode_budget,storage_bytes,actual_storage

class Contracts(unittest.TestCase):
 def test_factorized_reconstruction_matches_formula_and_gradients(self):
  import torch
  from efficient_adapter import reconstruct
  torch.manual_seed(7)
  a=torch.randint(0,2,(8,9)).double();v=torch.randn(2,9,3,dtype=torch.double,requires_grad=True);w=torch.randn(2,9,dtype=torch.double,requires_grad=True)
  expected=torch.einsum('ne,beh,be->bhn',a,v,w);actual=reconstruct(a,v,w)
  torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
  gv,gw=torch.autograd.grad(expected.square().sum(),(v,w),retain_graph=True)
  av,aw=torch.autograd.grad(actual.square().sum(),(v,w))
  torch.testing.assert_close(av,gv,rtol=1e-12,atol=1e-12);torch.testing.assert_close(aw,gw,rtol=1e-12,atol=1e-12)

 def test_local_swaps_preserve_incidence_margins_and_snapshots(self):
  n=32;edges=[sorted({i,(i+1)%n,(i+4)%n}) for i in range(n)]
  records=[dict(Fwy='R'+str(i%2),Direction='E' if i%2 else 'W') for i in range(n)]
  points=np.arange(n)*.005;d=np.abs(points[:,None]-points)
  result=local_chain(edges,d,records,19073)
  self.assertEqual(set(result),{.25,.5,1.})
  a,b=margins(edges,n)
  for e,m in result.values():
   aa,bb=margins(e,n);self.assertEqual(a,aa);np.testing.assert_array_equal(b,bb)
   self.assertEqual(len({tuple(x) for x in e}),len(e));self.assertGreater(m['changed_supports'],0)
  self.assertNotEqual(result[.25][0],result[1.][0])

 def test_exact_count_sampler_is_original_sampler(self):
  from memory_sampling import select_memory_origins
  origins=np.arange(12,20998,dtype=np.int64)
  expected=select_memory_origins(origins,12,96,horizon=12,train_end=21009,seq_len=12)
  np.testing.assert_array_equal(select_count(origins,len(expected)),expected)
  import hashlib
  self.assertEqual(hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest(),'c22849eef3de9a71c68c7a356f8833f2d957e5e513d6734750f0386cfd23158b')

 def test_byte_budget_is_measured_tensor_storage(self):
  import torch
  from memory_sampling import build_phase_balanced_bank
  for edges in [1,3,12]:
   m=17;bank=build_phase_balanced_bank(torch.arange(m),torch.randn(m,edges,13),torch.randn(m,edges,12),'cpu')
   self.assertEqual(actual_storage(bank)['unique_tensor_bytes'],storage_bytes(m,edges))
  for edges in [685,716,10000]:
   count,target=episode_budget(edges);self.assertLessEqual(storage_bytes(count,edges),target)
   self.assertGreater(storage_bytes(count+1,edges),target)

 def test_prepared_data_excludes_test_origins(self):
  from common import arrays
  a=arrays();self.assertNotIn('origins_test',a);self.assertEqual(len(a['data']),28036)
  self.assertEqual(len(a['origins_val'])//64*64,6976)

if __name__=='__main__':unittest.main(verbosity=2)
