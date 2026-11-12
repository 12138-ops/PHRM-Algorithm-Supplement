"""Profile the exact frozen LargeST-SD STID/STGformer production pairs."""
import gc, os, sys, time
import numpy as np
import torch
from common import ROOT,PROJECTS,write_json,sha,log
from evidence import timings,gpu_activity


@torch.inference_mode()
def profile(backbone,mode,device):
    assert backbone in ('stid','stgformer') and mode in ('base','phrm')
    if gpu_activity():raise RuntimeError('GPU is shared; postpone efficiency measurements')
    sys.path.insert(0,str(ROOT/'largest'))
    from strong_baselines import build_model
    from physical_graph import load_graph
    from physics_kv import StrongBackbonePhysicsKV,periodic_prior,residual_history,residual_future,edge_descriptors
    from memory_sampling import select_memory_origins,build_phase_balanced_bank
    from structure_controls import memory_storage
    old=PROJECTS/'largestsd_strong_kv_3seed_20261015'
    prepared=old/'data/processed/LargeSTSD_2019_15min_l12_h12.npz'
    with np.load(prepared,allow_pickle=False) as z:
        data=z['data'];node_ids=z['node_ids'];origins_train=z['origins_train'].astype('int64')+1;origins_val=z['origins_val'].astype('int64')+1
    assert data.shape==(35040,716)
    base=build_model(backbone,old,device);ck=old/f'runs/{backbone}/seed2024/model.pt'
    payload=torch.load(ck,map_location='cpu',weights_only=False);base.load_state_dict(payload['model'],strict=True);base.eval()
    graph=load_graph(old/'data/physical/sd_rn_adj.npy',old/'data/physical/sd_meta.csv',expected_sensor_ids=node_ids)
    cutoff=int(origins_val[0])-12-1;assert cutoff==21009
    st=time.monotonic();daily,weekly=periodic_prior(data[:cutoff],96,672);torch.cuda.synchronize();periodic_seconds=time.monotonic()-st;daily=daily.to(device);weekly=weekly.to(device)
    chosen=select_memory_origins(origins_train[origins_train+12<=cutoff],12,96,horizon=12,train_end=cutoff,seq_len=12)
    st=time.monotonic();pool=graph['edge_pool'].to(device);keys=[];values=[]
    for start in range(0,len(chosen),32):
        oo=chosen[start:start+32];o=torch.as_tensor(oo,device=device);x=torch.from_numpy(np.stack([data[t-12:t] for t in oo])).to(device);y=torch.from_numpy(np.stack([data[t:t+12] for t in oo])).to(device)
        keys.append(edge_descriptors(residual_history(x,o,daily,weekly),o,pool,period=96).cpu());values.append(torch.einsum('bhn,ne->beh',residual_future(y,o,daily,weekly),pool).cpu())
    bank=build_phase_balanced_bank(torch.as_tensor(chosen),torch.cat(keys),torch.cat(values),device,8,.5);torch.cuda.synchronize();bank_seconds=time.monotonic()-st
    adapter_ck=old/f'runs/{backbone}/seed2024/adapter.pt';saved=torch.load(adapter_ck,map_location='cpu',weights_only=False)
    model=StrongBackbonePhysicsKV(base,graph,daily,weekly,12,96).to(device);state=model.state_dict();state.update(saved['adapter']);model.load_state_dict(state,strict=True);model.eval();weight=torch.as_tensor(saved['horizon_gate']['weights'],device=device)
    storage=memory_storage(bank);adapter_params=sum(p.numel() for p in model.parameters() if p.requires_grad);base_params=sum(p.numel() for p in base.parameters());profiles=[]
    if mode=='base':del model,bank,pool,daily,weekly;gc.collect();torch.cuda.empty_cache()
    del keys,values
    for batch in (1,32):
        oo=origins_val[:batch];x=torch.from_numpy(np.stack([data[t-12:t] for t in oo])).to(device);o=torch.as_tensor(oo,device=device)
        fn=(lambda:base(x,o)) if mode=='base' else (lambda:model(x,o,bank,residual_weight=weight))
        stats=timings(fn);stats.update(batch_size=batch,windows_per_second=1000*batch/stats['median_wall_ms']);profiles.append(stats)
    result=dict(dataset='LargeST-SD',backbone=backbone,mode=mode,horizon=12,seq_len=12,seed=2024,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),device=torch.cuda.get_device_name(),pytorch=torch.__version__,cuda=torch.version.cuda,profiles=profiles,offline_periodic_seconds=periodic_seconds if mode!='base' else 0,offline_bank_seconds=bank_seconds if mode!='base' else 0,bank_storage=storage if mode!='base' else None,memory_episodes=len(chosen),memory_max_target_end=int(chosen[-1]+12),memory_fit_end=cutoff,base_parameters=base_params,trainable_adapter_parameters=adapter_params if mode!='base' else 0,source_base_checkpoint_sha256=sha(ck),source_adapter_sha256=sha(adapter_ck),source_processed_sha256=sha(prepared),cache_used_for_inference=False,validation_inputs_only=True,external_gpu_processes_after=gpu_activity())
    assert not result['external_gpu_processes_after'],'concurrent process invalidated efficiency profile'
    write_json(ROOT/'efficiency'/f'LargeSTSD_{backbone}_{mode}.json',result);log('EFFICIENCY_RESULT',dataset='LargeST-SD',backbone=backbone,mode=mode,median_ms=[p['median_wall_ms'] for p in profiles])
