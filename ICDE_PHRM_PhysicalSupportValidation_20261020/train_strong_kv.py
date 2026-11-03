"""Validation-only paired KV attachment to pinned, frozen strong backbones."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiment import (prepared, loader, create_bank, evaluate, calibrate, sha,
                        write_json, save_checkpoint, HORIZON, LENGTH, PERIOD)
from evaluate_locked import restore_strong, assert_metrics_match
from physical_graph import load_graph
from physics_kv import StrongBackbonePhysicsKV, periodic_prior, trainable_parameters
from metrics import masked_mae_loss
from strong_baselines import model_cfg

VERSION='largest-strong-backbone-kv-validation-v1'
DEVICES={'stid':2,'stgformer':3}
RECIPE=dict(seed=2024,seq_len=12,horizon=12,hidden=96,optimizer='AdamW',lr=.002,
    weight_decay=.0001,batch_size=32,max_epochs=30,patience=5,clip_grad=5.,
    loss='raw_zero_masked_mae',scale_regularization=.0001,
    memory_stride_budget=12,topk=8,temperature=.5,
    calibration_grid=np.linspace(0,1,9).tolist(),selection='uncalibrated full validation raw masked MAE')


def check_recipe(root):
    path=root/'audit/launch_manifest.json'
    if not path.is_file():raise RuntimeError('missing frozen setup manifest')
    manifest=json.loads(path.read_text())
    if manifest['recipe']!=RECIPE:raise ValueError('training recipe differs from frozen setup')
    for rel,digest in manifest['files'].items():
        if sha(root/rel)!=digest:raise RuntimeError(f'fixed input changed: {rel}')
    return manifest


def capture_base(base):
    return {key:value.detach().cpu().clone() for key,value in
            list(base.named_parameters())+list(base.named_buffers())}


def check_base(base, frozen):
    actual=dict(list(base.named_parameters())+list(base.named_buffers()))
    if set(actual)!=set(frozen):raise RuntimeError('baseline parameter/buffer keys changed')
    for key,reference in frozen.items():
        torch.testing.assert_close(actual[key].detach().cpu(),reference,rtol=0,atol=0)
    if base.training or any(p.requires_grad or p.grad is not None for p in base.parameters()):
        raise RuntimeError('backbone was not kept frozen in eval mode')


def setup(root,name,device):
    arrays,protocol,_=prepared(root)
    if 'origins_test' in arrays:raise RuntimeError('validation task exposes test origins')
    base,audit=restore_strong(root,arrays,protocol,name,2024,device)
    graph=load_graph(root/'data/physical/sd_rn_adj.npy',root/'data/physical/sd_meta.csv',
                     expected_sensor_ids=arrays['node_ids'])
    cutoff=int(arrays['origins_val'][0])-LENGTH-1
    if cutoff!=21009:raise RuntimeError('training memory cutoff changed')
    daily,weekly=periodic_prior(arrays['data'][:cutoff],PERIOD,PERIOD*7)
    model=StrongBackbonePhysicsKV(base,graph,daily.to(device),weekly.to(device),HORIZON,96).to(device)
    bank,sampling=create_bank(arrays,graph,model.daily,model.weekly,device)
    if sampling['memory_episodes']!=1749:raise RuntimeError('frozen memory budget changed')
    if 'origins_test' in arrays:raise RuntimeError('test origins must remain inaccessible')
    return model,bank,arrays,protocol,graph,sampling,audit


def preflight(root,name,device,physical_gpu):
    model,bank,arrays,protocol,graph,sampling,audit=setup(root,name,device)
    frozen=capture_base(model.base)
    x,y,t=next(iter(loader(arrays,arrays['origins_val'])))
    x=x.to(device);y=y.to(device);t=t.to(device)
    model.eval()
    with torch.no_grad():
        pred,info=model(x,t,bank)
        torch.testing.assert_close(pred,info['base'],rtol=0,atol=0)
        torch.testing.assert_close(pred,model.base(x,t),rtol=0,atol=0)
    # The production train batch and a real optimizer step exercise all paths.
    model.train()
    if model.base.training:raise RuntimeError('dropout enabled in frozen backbone')
    opt=torch.optim.AdamW(trainable_parameters(model),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    pred,_=model(x[:32],t[:32],bank)
    objective=masked_mae_loss(pred,y[:32],float(arrays['mean']),float(arrays['std']))
    if not torch.isfinite(objective):raise RuntimeError('nonfinite preflight objective')
    objective.backward();opt.step();check_base(model.base,frozen)
    with torch.no_grad():
        pred,info=model(x,t,bank,disable_memory=True)
        torch.testing.assert_close(pred,info['base'],rtol=0,atol=0)
    report=dict(status='PASS',model=name,physical_gpu=physical_gpu,seed=2024,
        validation_restore=audit['validation_restore'],sampling=sampling,
        zero_start_equals_baseline=True,disable_memory_equals_baseline=True,
        finite_backward=True,base_parameters_and_buffers_unchanged=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        test_evaluated=False,test_labels_used_for_selection=False)
    write_json(root/f'audit/{name}_preflight.json',report)
    print('STRONG_KV_PREFLIGHT '+json.dumps(report),flush=True)


def train(root,name,device,physical_gpu):
    result_path=root/f'results/{name}_kv_validation_seed2024.json'
    output=root/f'runs/{name}/seed2024/adapter.pt'
    if result_path.exists() or output.exists():raise FileExistsError('do not overwrite completed or interrupted runs')
    preflight_record=json.loads((root/f'audit/{name}_preflight.json').read_text())
    if preflight_record.get('status')!='PASS' or preflight_record['physical_gpu']!=physical_gpu:
        raise RuntimeError('matching real-data preflight required')
    model,bank,arrays,protocol,graph,sampling,base_audit=setup(root,name,device)
    checkpoint=root/f'runs/{name}/seed2024/model.pt'
    base_hash=sha(checkpoint)
    frozen=capture_base(model.base)
    baseline=evaluate(model,arrays,arrays['origins_val'],device,bank)
    assert_metrics_match(baseline['base'],baseline['kv'],'zero-start baseline')
    source=json.loads((root/f'results/{name}_validation_seed2024.json').read_text())
    assert_metrics_match(baseline['base'],source['metrics'],'copied source baseline')
    opt=torch.optim.AdamW(trainable_parameters(model),lr=RECIPE['lr'],weight_decay=RECIPE['weight_decay'])
    best=float('inf');best_epoch=bad=0;best_state=None;epoch_rows=[]
    print('STRONG_KV_SETUP '+json.dumps(dict(model=name,recipe=RECIPE,base_sha256=base_hash,
        sampling=sampling,physical_gpu=physical_gpu,base_configuration=model_cfg(name))),flush=True)
    for epoch in range(1,RECIPE['max_epochs']+1):
        model.train();tick=last=time.monotonic();total=steps=0
        for x,y,t in loader(arrays,arrays['origins_train'],32,True):
            x=x.to(device);y=y.to(device);t=t.to(device);opt.zero_grad(set_to_none=True)
            pred,_=model(x,t,bank)
            loss=masked_mae_loss(pred,y,float(arrays['mean']),float(arrays['std']))
            loss=loss+RECIPE['scale_regularization']*torch.tanh(model.residual_scale).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite adapter loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(trainable_parameters(model),RECIPE['clip_grad']);opt.step()
            total+=float(loss.detach());steps+=1
            if time.monotonic()-last>=30:
                print(f'STRONG_KV_PROGRESS model={name} epoch={epoch} batch={steps}',flush=True);last=time.monotonic()
        scores=evaluate(model,arrays,arrays['origins_val'],device,bank)
        value=scores['kv']['raw_masked']['mae'];check_base(model.base,frozen)
        row=dict(epoch=epoch,train_mae=total/steps,validation=scores['kv']['horizon_average'],seconds=time.monotonic()-tick)
        epoch_rows.append(row)
        print(f'STRONG_KV_EPOCH model={name} epoch={epoch} train_mae={total/steps:.6f} '
            f'val_mae={value:.6f} val_rmse={scores["kv"]["horizon_average"]["rmse"]:.6f} '
            f'base_val_mae={baseline["base"]["raw_masked"]["mae"]:.6f} seconds={row["seconds"]:.1f}',flush=True)
        if value<best:
            best=value;best_epoch=epoch;bad=0
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items() if not key.startswith('base.')}
        else:bad+=1
        if bad>=RECIPE['patience']:print('STRONG_KV_EARLY_STOP',flush=True);break
    if best_state is None:raise RuntimeError('no valid adapter checkpoint')
    state=model.state_dict();state.update(best_state);model.load_state_dict(state,strict=True)
    uncalibrated=evaluate(model,arrays,arrays['origins_val'],device,bank)
    calibration=calibrate(model,arrays,arrays['origins_val'],bank,device)
    scores=evaluate(model,arrays,arrays['origins_val'],device,bank,calibration['weights'])
    check_base(model.base,frozen)
    if sha(checkpoint)!=base_hash:raise RuntimeError('baseline checkpoint bytes changed')
    assert_metrics_match(scores['base'],source['metrics'],'final paired baseline')
    gains={stage:{metric:1-values['kv']['horizon_average'][metric]/values['base']['horizon_average'][metric]
        for metric in ('mae','mse','rmse','mape_percent')}
        for stage,values in [('uncalibrated',uncalibrated),('calibrated',scores)]}
    decision=('UNCALIBRATED_MAE_RMSE_GAIN' if gains['uncalibrated']['mae']>0 and gains['uncalibrated']['rmse']>0
              else 'CALIBRATION_ONLY_OR_MIXED_GAIN' if gains['calibrated']['mae']>0 else 'NO_VALIDATION_MAE_GAIN')
    result=dict(version=VERSION,dataset='LargeST-SD',model=name+'+PhysicsResidualKV',seed=2024,
        split='validation',seq_len=12,horizon=12,recipe=RECIPE,protocol=protocol,metrics=scores,
        uncalibrated=uncalibrated,relative_gain=gains,best_epoch=best_epoch,epochs=epoch_rows,
        horizon_gate=calibration,sampling=sampling,normalization=bank.normalization_metadata,
        physical_hypergraph=graph['metadata'],base_checkpoint=str(checkpoint),base_checkpoint_sha256=base_hash,
        paired_same_checkpoint=True,base_parameters_and_buffers_unchanged=True,base_weights_unchanged=True,
        validation_restore=base_audit['validation_restore'],test_evaluated=False,test_labels_used_for_selection=False,
        memory_train_only=True,memory_template_fit_end=21009,physical_gpu=physical_gpu,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,screen_decision=decision,
        interpretation='single-seed validation screen; selection validation is not independent test or cross-seed stability evidence',
        smoke=False)
    save_checkpoint(output,dict(adapter=best_state,seed=2024,protocol=protocol,recipe=RECIPE,
        base_checkpoint_sha256=base_hash,best_epoch=best_epoch,horizon_gate=calibration,
        sampling=sampling,normalization=bank.normalization_metadata,version=VERSION,smoke=False))
    check_recipe(root)
    write_json(result_path,result)
    print('STRONG_KV_VALIDATION_RESULT '+json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.')
    parser.add_argument('--model',choices=DEVICES,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    gpu=DEVICES[args.model]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError(f'{args.model} owns physical GPU{gpu} only')
    root=Path(args.root).resolve()
    if root!=Path(__file__).resolve().parent:raise RuntimeError('new isolated project required')
    check_recipe(root)
    if (root/'audit/test_release.json').exists():raise RuntimeError('this project must stay validation-only')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    random.seed(2024);np.random.seed(2024);torch.manual_seed(2024);torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    if args.preflight_only:preflight(root,args.model,device,gpu)
    else:train(root,args.model,device,gpu)


if __name__=='__main__':main()
