"""Matched-policy replay, end-to-end profiling and actual retrieval evidence."""
import argparse, csv, gc, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, PROJECTS, sha, write_json, log, require_gpu
from ablations import setup, run as fit, evaluate_test
from structure_controls import memory_storage
import frozen_physics as physics


def inputs():
    return json.loads((ROOT/'audit/input_manifest.json').read_text())


def main_job(ds, h=96, seed=2024):
    row=inputs()['main_records'][f'{ds}_h{h}_s{seed}']
    return dict(dataset=ds,horizon=h,seed=seed,variant='full',
                source_result=row['path'],source_result_sha256=row['sha256'],
                source_adapter=str(Path(row['path']).parent/'adapter.pt'),source_family='main_paired')


def restore(job,device):
    data,model,bank,graph,base,bank_o=setup(job,device)
    source=Path(job['source_result']);assert sha(source)==job['source_result_sha256']
    r=json.loads(source.read_text());ck=Path(job['source_adapter'])
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    current=model.state_dict();state=saved['adapter']
    assert not any(k.startswith('base.') for k in state)
    assert set(state)=={k for k in current if not k.startswith('base.')},'adapter keys differ'
    for key in ('incidence','edge_pool'):
        if key in state:assert torch.equal(current[key].cpu(),state[key].cpu()),key+' mismatch'
    for key in ('daily','weekly'):
        if key in state:assert torch.allclose(current[key].cpu(),state[key].cpu(),atol=2e-6,rtol=2e-6),key+' mismatch'
    current.update(state);model.load_state_dict(current,strict=True);model.eval()
    assert str(base)==r['base_checkpoint']
    return data,model,bank,graph,r,saved


@torch.inference_mode()
def calibrate(data,model,bank,h,device):
    """Only validation labels enter the fixed nine-point per-step grid."""
    grid=torch.linspace(0,1,9,device=device);scores=torch.zeros(h,9,device=device,dtype=torch.float64);abscores=torch.zeros_like(scores)
    bse=bae=rse=rae=0.;count=perstep=0;batch=16 if data.metadata['dataset']=='PEMS08' else 4
    origins=data.origins(h,'val')
    for start in range(0,len(origins),batch):
        oo=origins[start:start+batch];o=torch.as_tensor(oo,device=device)
        x=torch.from_numpy(np.stack([data.values[t-96:t] for t in oo])).to(device)
        y=torch.from_numpy(np.stack([data.values[t:t+h] for t in oo])).to(device)
        b,d,f=model.decompose(x,o,bank);_,delta=model.mix(b,d,f);be=b-y;re=be+delta
        bse+=float(be.square().sum(dtype=torch.float64));bae+=float(be.abs().sum(dtype=torch.float64));rse+=float(re.square().sum(dtype=torch.float64));rae+=float(re.abs().sum(dtype=torch.float64));count+=y.numel();perstep+=len(y)*y.shape[-1]
        for j,w in enumerate(grid):
            e=be+w*delta;scores[:,j]+=e.square().sum((0,2),dtype=torch.float64);abscores[:,j]+=e.abs().sum((0,2),dtype=torch.float64)
    ix=scores.argmin(1);selected=grid[ix]
    metrics=dict(trained=dict(mse=float(scores.gather(1,ix[:,None]).sum()/count),mae=float(abscores.gather(1,ix[:,None]).sum()/count)),base=dict(mse=bse/count,mae=bae/count))
    raw=dict(trained=dict(mse=rse/count,mae=rae/count),base=metrics['base'])
    return selected,metrics,raw,dict(grid=grid.cpu().tolist(),weights=selected.cpu().tolist(),selection_split='validation',test_labels_used_for_selection=False,windows=len(origins))


def replay(job,device):
    start=time.monotonic();data,model,bank,graph,source,saved=restore(job,device)
    w,vm,raw,cal=calibrate(data,model,bank,job['horizon'],device)
    if job['source_family']=='main_paired':
        expected=source['validation']['trained']
        assert abs(vm['trained']['mse']-expected['mse'])<2e-5,(job,vm,expected)
        savedweights=source['horizon_gate']['weights']
        assert np.allclose(w.cpu().numpy(),savedweights,atol=1e-7),'main calibration replay differs'
    path=ROOT/'results'/job['id'];path.mkdir(parents=True,exist_ok=True)
    release=dict(source_adapter=job['source_adapter'],source_adapter_sha256=sha(job['source_adapter']),base_checkpoint_sha256=sha(source['base_checkpoint']),weights=w.cpu().tolist(),validation=vm,raw_validation=raw,calibration=cal,test_labels_used_for_selection=False,policy='matched main-table validation grid; checkpoint frozen before replay')
    write_json(path/'test_release.json',release)
    tm=evaluate_test(model,data,bank,job['horizon'],16 if job['dataset']=='PEMS08' else 4,device,w)
    assert sha(job['source_adapter'])==release['source_adapter_sha256']
    out=dict(job=job,status='COMPLETE',split='test',dataset=job['dataset'],horizon=job['horizon'],seed=job['seed'],variant=job['variant'],protocol=data.metadata,validation=vm,raw_validation=raw,test=tm,calibration=cal,applied_weights=w.cpu().tolist(),graph=graph['metadata'],base_checkpoint=source['base_checkpoint'],memory_train_only=True,test_labels_used_for_selection=False,source_release=release,seconds=time.monotonic()-start)
    write_json(path/'result.json',out);log('ALIGNED_RESULT',job=job['id'],validation=vm['trained'],test=tm['trained'],seconds=out['seconds'])


def gpu_activity():
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    result=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return [line for line in result.splitlines() if line.strip() and int(line.split(',')[0])!=os.getpid()]


@torch.inference_mode()
def timings(fn,warmup=10,repeats=50):
    for _ in range(warmup):fn()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated();wall=[];events=[]
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();torch.cuda.synchronize()
        wall.append((time.perf_counter()-start)*1000);events.append(a.elapsed_time(b))
    return dict(warmup=warmup,repeats=repeats,wall_ms=wall,cuda_event_ms=events,median_wall_ms=float(np.median(wall)),p95_wall_ms=float(np.quantile(wall,.95)),median_cuda_ms=float(np.median(events)),allocated_before_mib=baseline/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_increment_mib=(torch.cuda.max_memory_allocated()-baseline)/2**20)


def efficiency(ds,mode,device):
    """Each process profiles one method; all retrieval work remains online."""
    if gpu_activity():raise RuntimeError('GPU is shared; postpone efficiency measurements')
    begin=time.monotonic();job=main_job(ds);data,model,bank,graph,r,_=restore(job,device);torch.cuda.synchronize();build_restore=time.monotonic()-begin
    # Measure clean offline bank construction separately from model restore.
    from ablation_model import build_bank
    st=time.monotonic();daily,weekly=physics.periodic_prior(np.array(data.values[:data.metadata['train_end']],copy=True),288,2016);torch.cuda.synchronize();period_seconds=time.monotonic()-st
    origins=data.origins(96,'train')[::12]
    del bank;gc.collect();torch.cuda.empty_cache()
    st=time.monotonic()
    bank=build_bank(data,origins,96,graph,daily.to(device),weekly.to(device),device,32 if ds=='PEMS08' else 4,'full');torch.cuda.synchronize();bank_seconds=time.monotonic()-st
    weight=torch.as_tensor(r['horizon_gate']['weights'],device=device);profiles=[]
    if mode=='base':
        base=model.base;del model,bank;gc.collect();torch.cuda.empty_cache()
    for batch in (1,16 if ds=='PEMS08' else 4):
        oo=data.origins(96,'val')[:batch];x=torch.from_numpy(np.stack([data.values[t-96:t] for t in oo])).to(device);o=torch.as_tensor(oo,device=device)
        fn=(lambda:base(x,o%288)) if mode=='base' else (lambda:physics.TQNetPhysicsKV.forward(model,x,o,bank,residual_weight=weight))
        profile=timings(fn);profile.update(batch_size=batch,windows_per_second=1000*batch/profile['median_wall_ms']);profiles.append(profile)
    result=dict(dataset=ds,mode=mode,horizon=96,seq_len=96,seed=2024,device=torch.cuda.get_device_name(),physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),pytorch=torch.__version__,cuda=torch.version.cuda,source_adapter=job['source_adapter'],source_adapter_sha256=sha(job['source_adapter']),source_result_sha256=job['source_result_sha256'],profiles=profiles,offline_periodic_seconds=period_seconds if mode!='base' else 0,offline_bank_seconds=bank_seconds if mode!='base' else 0,setup_and_restore_seconds=build_restore,bank_storage=None if mode=='base' else memory_storage(bank),base_parameters=sum(p.numel() for p in (base if mode=='base' else model.base).parameters()),trainable_adapter_parameters=0 if mode=='base' else sum(p.numel() for p in model.parameters() if p.requires_grad),cache_used_for_inference=False,validation_inputs_only=True,external_gpu_processes_after=gpu_activity())
    assert not result['external_gpu_processes_after'],'concurrent GPU task invalidated efficiency profile'
    write_json(ROOT/'efficiency'/f'{ds}_{mode}.json',result);log('EFFICIENCY_RESULT',dataset=ds,mode=mode,median_ms=[p['median_wall_ms'] for p in profiles])


@torch.inference_mode()
def cases(ds,device):
    job=main_job(ds);data,model,bank,graph,r,_=restore(job,device);weight=torch.as_tensor(r['horizon_gate']['weights'],device=device)
    # Fixed chronological probe, then one helpful and one harmful case. Selection
    # is descriptive and never changes a learned weight or reported aggregate.
    all_o=data.origins(96,'val');probes=all_o[np.linspace(0,len(all_o)-1,96,dtype=int)];rows=[]
    for t in probes:
        x=torch.from_numpy(data.values[t-96:t].copy())[None].to(device);o=torch.tensor([t],device=device);y=torch.from_numpy(data.values[t:t+96].copy())[None].to(device)
        b,d,f=model.decompose(x,o,bank);pred,_=model.mix(b,d,f,weight);base_error=(b-y).square().mean(1)[0];new_error=(pred-y).square().mean(1)[0]
        utility=base_error-new_error
        rows.append(dict(origin=int(t),mean_utility=float(utility.mean()),best_node=int(utility.argmax()),worst_node=int(utility.argmin())))
    picked=[('helpful',max(rows,key=lambda z:z['mean_utility'])),('harmful',min(rows,key=lambda z:z['mean_utility']))]
    outdir=ROOT/'cases'/ds;outdir.mkdir(parents=True,exist_ok=True)
    write_json(outdir/'probe_selection.json',dict(selection_split='validation',selection_rule='fixed 96 evenly spaced origins; extreme mean correction utilities for descriptive contrast only',checkpoint_or_hyperparameter_selection=False,rows=rows))
    for label,row in picked:
        t=row['origin'];node=row['best_node'] if label=='helpful' else row['worst_node'];o=torch.tensor([t],device=device);x=torch.from_numpy(data.values[t-96:t].copy())[None].to(device)
        b,d,f=model.decompose(x,o,bank);pred,_=model.mix(b,d,f,weight)
        q=physics.edge_descriptors(physics.residual_history(x,o,model.daily,model.weekly),o,model.edge_pool)
        retrieval,_,support,nearest=bank.retrieve(q,o,96)
        importance=support[0]/(1+nearest[0,:,0]);incident=torch.where(model.incidence[node]>0)[0];edge=int(incident[importance[incident].argmax()]);members=graph['edges'][edge]
        normalized=(q-bank.mean)/bank.std
        # Exact production distance and chronological exclusion.
        query=normalized[:,edge:edge+1,:];keys=bank.keys[:,edge:edge+1,:]
        distance=(query.square().sum(-1)[:,:,None]+keys.square().sum(-1).T[None,:,:]-2*torch.einsum('bed,med->bem',query,keys)).clamp_min(0)
        distance=distance.masked_fill(~(bank.origins[None,None,:]+96<=o[:,None,None]-96),float('inf'))
        distances,ids=distance.topk(min(bank.topk,len(bank.origins)),largest=False);valid=torch.isfinite(distances);w=torch.softmax(-distances.masked_fill(~valid,1e6)/bank.temperature,-1)*valid;w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        ids=ids[0,0];w=w[0,0];trajectories=bank.values[ids,edge];reconstructed=(w[:,None]*trajectories).sum(0)
        assert torch.allclose(reconstructed,retrieval[0,edge],atol=3e-5,rtol=3e-5),'retrieval tracing mismatch'
        origins=bank.origins[ids].cpu().numpy();assert np.all(origins+96<=t-96)
        history=physics.residual_history(x,o,model.daily,model.weekly)[0,:,members].mean(-1).cpu().numpy()
        path=outdir/(label+'.npz')
        np.savez(path,origin=t,node=node,edge=edge,members=members,history=data.values[t-96:t,node],query_edge_residual_history=history,truth=data.values[t:t+96,node],base=b[0,:,node].cpu().numpy(),phrm=pred[0,:,node].cpu().numpy(),memory_candidate=(b+d)[0,:,node].cpu().numpy(),memory_origins=origins,retrieval_weights=w.cpu().numpy(),retrieved_edge_residual_values=trajectories.cpu().numpy(),weighted_edge_residual=reconstructed.cpu().numpy())
        report=dict(label=label,dataset=ds,split='validation',origin=t,node=node,support_edge_index=edge,members=members,selection=row,memory_origins=origins.tolist(),weights=w.cpu().tolist(),nearest_squared_distances=distances[0,0].cpu().tolist(),eligibility_rule='memory_origin + horizon <= query_origin - seq_len',chronology_pass=True,source_result_sha256=job['source_result_sha256'],source_adapter_sha256=sha(job['source_adapter']),data_protocol=data.metadata,npz_sha256=sha(path),stored_value_type='pooled hyperedge residual future; not individual node values',physical_coordinates_available=False,causal_effect_claimed=False)
        write_json(outdir/(label+'.json'),report)
        with (outdir/(label+'_retrieval.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['rank','memory_origin','memory_target_end','query_history_start','weight','squared_distance'])
            writer.writerows((i+1,int(u),int(u+96),t-96,float(ww),float(dd)) for i,(u,ww,dd) in enumerate(zip(origins,w.cpu(),distances[0,0].cpu())))
    write_json(outdir/'graph.json',dict(edges=graph['edges'],metadata=graph['metadata']));log('REAL_CASES_COMPLETE',dataset=ds,cases=2,selection_split='validation')


def smoke(device):
    job=dict(dataset='PEMS08',horizon=24,seed=2024,variant='node_independent_memory')
    data,model,bank,graph,_,origins=setup(job,device)
    t=int(data.origins(24,'val')[0]);x=torch.from_numpy(data.values[t-96:t].copy())[None].to(device);o=torch.tensor([t],device=device);y=torch.from_numpy(data.values[t:t+24].copy())[None].to(device)
    pred,_=model(x,o,bank);assert torch.isfinite(pred).all();loss=(pred-y).square().mean();loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum()>0) for p in model.parameters() if p.requires_grad)
    assert bool((bank.origins+24<=data.metadata['train_end']).all())
    assert graph['incidence'].shape==(170,170)
    write_json(ROOT/'audit/gpu_preflight.json',dict(passed=True,physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),loss=float(loss),memory_episodes=len(origins),finite_backward=True,node_coverage=170,train_only=True))
    log('PREFLIGHT_PASS',loss=float(loss))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['fit','replay','efficiency','cases','smoke','largest_efficiency']);p.add_argument('--index',type=int);p.add_argument('--dataset',choices=['PEMS08','PEMS07']);p.add_argument('--mode',default='phrm');p.add_argument('--backbone',default='stid');a=p.parse_args();_,device=require_gpu();torch.set_num_threads(2)
    if a.task=='smoke':smoke(device)
    elif a.task in ('fit','replay'):
        job=inputs()['jobs'][a.index];assert job['kind']==a.task
        (fit if a.task=='fit' else replay)(job,device)
    elif a.task=='efficiency':efficiency(a.dataset,a.mode,device)
    elif a.task=='cases':cases(a.dataset,device)
    elif a.task=='largest_efficiency':
        from largest_evidence import profile
        profile(a.backbone,a.mode,device)
