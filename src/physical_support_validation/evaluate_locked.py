"""One-way test evaluation after all validation choices have been frozen.

The release manifest covers every model/seed, not merely the run requested by
the CLI. No calibration, checkpoint selection, or memory update occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import random
import tempfile

import numpy as np
import torch

from experiment import (EVAL_BATCH, HORIZON, LENGTH, PERIOD, SEEDS, VERSION,
                        config, create_bank, evaluate, loader, prepared, sha)
from metrics import Metrics
from models.tqnet import Model as TQNet
from physical_graph import load_graph
from physics_kv import TQNetPhysicsKV, periodic_prior


MODELS = ('tqnet_kv', 'stid', 'stgformer')
RELEASE = 'audit/test_release.json'
CORE_FILES = (
    'experiment.py', 'evaluate_locked.py', 'protocol_data.py', 'metrics.py',
    'memory_sampling.py', 'physics_kv.py', 'physical_graph.py',
    'models/tqnet.py', 'strong_baselines.py',
    'data/processed/LargeSTSD_2019_15min_l12_h12.npz',
    'data/processed/protocol.json', 'data/physical/sd_rn_adj.npy',
    'data/physical/sd_meta.csv',
)


def required_release_files():
    paths = set(CORE_FILES)
    for seed in SEEDS:
        paths.update((f'runs/seed{seed}/baseline.pt', f'runs/seed{seed}/adapter.pt',
                      f'results/baseline_validation_seed{seed}.json',
                      f'results/paired_validation_seed{seed}.json'))
        for name in ('stid', 'stgformer'):
            paths.add(f'runs/{name}/seed{seed}/model.pt')
            paths.add(f'results/{name}_validation_seed{seed}.json')
    return paths


def validate_hashes(root, manifest=None):
    """Fail closed unless the complete frozen comparison is still intact."""
    root = Path(root)
    release_path = root / RELEASE
    if not release_path.is_file():
        raise RuntimeError('test remains locked: missing complete validation release')
    if manifest is None:
        manifest = json.loads(release_path.read_text(encoding='utf-8'))
    if tuple(manifest.get('seeds', ())) != SEEDS:
        raise ValueError('release must freeze all three paired seeds')
    names = manifest.get('models', ())
    if len(names) != len(MODELS) or set(names) != set(MODELS):
        raise ValueError('release must freeze TQNet/KV, STID and STGformer')
    if manifest.get('test_labels_used_for_selection') is not False:
        raise ValueError('test labels may not be used for selection')
    if manifest.get('primary_metric') != 'horizon_average':
        raise ValueError('official primary metrics must average forecast steps')
    files = manifest.get('files')
    if not isinstance(files, dict) or not required_release_files().issubset(files):
        missing = sorted(required_release_files() - set(files or ()))
        raise ValueError(f'incomplete frozen comparison manifest: {missing}')
    for relative, expected in files.items():
        if not isinstance(relative, str) or '\\' in relative:
            raise ValueError('release paths must be POSIX relative paths')
        path = Path(relative)
        if path.is_absolute() or PureWindowsPath(relative).drive or '..' in path.parts:
            raise ValueError(f'unsafe release path: {relative}')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f'invalid SHA256 entry: {relative}')
        target = root / path
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f'frozen release hash mismatch: {relative}')
    return manifest


def validate_checkpoint_metadata(payload, protocol, seed, expected_config=None):
    if payload.get('protocol') != protocol or payload.get('seed') != seed:
        raise ValueError('checkpoint protocol/seed differs from the frozen run')
    if payload.get('smoke') is not False:
        raise ValueError('smoke or unlabelled checkpoint cannot enter the main comparison')
    if not isinstance(payload.get('best_epoch'), int) or payload['best_epoch'] < 1:
        raise ValueError('checkpoint has no valid validation-selected epoch')
    if expected_config is not None and payload.get('config') != expected_config:
        raise ValueError('checkpoint model/training configuration changed')


def validate_validation_record(record, protocol, seed):
    if record.get('split') != 'validation' or record.get('seed') != seed:
        raise ValueError('expected the matching validation-only selection record')
    if record.get('protocol') != protocol or record.get('smoke') is not False:
        raise ValueError('validation protocol changed or smoke run supplied')
    if record.get('test_labels_used_for_selection') is not False:
        raise ValueError('validation record does not exclude test-label selection')
    if record.get('test_evaluated') is not False:
        raise ValueError('selection record must precede test evaluation')


def frozen_gate(calibration):
    """Read an already selected gate. This function never receives labels."""
    if calibration.get('selection_split') != 'validation':
        raise ValueError('gate was not selected on validation')
    if calibration.get('test_labels_used_for_selection') is not False:
        raise ValueError('gate may have used test labels')
    weights = np.asarray(calibration.get('weights'), dtype=np.float64)
    grid = np.asarray(calibration.get('grid'), dtype=np.float64)
    if weights.shape != (HORIZON,) or not np.isfinite(weights).all():
        raise ValueError('invalid frozen horizon gate')
    if grid.shape != (9,) or not np.array_equal(grid, np.linspace(0, 1, 9)):
        raise ValueError('gate calibration grid differs from the frozen recipe')
    if not np.isin(weights, grid).all():
        raise ValueError('gate weights are outside the frozen validation grid')
    return weights.tolist()


def restore_adapter_state(model, adapter):
    """Require every non-base parameter/buffer and prohibit base replacement."""
    expected = {key for key in model.state_dict() if not key.startswith('base.')}
    if set(adapter) != expected:
        raise ValueError(f'adapter state mismatch; missing={sorted(expected-set(adapter))}; '
                         f'extra={sorted(set(adapter)-expected)}')
    if any(torch.is_floating_point(value) and not torch.isfinite(value).all()
           for value in adapter.values()):
        raise ValueError('nonfinite adapter state')
    merged = model.state_dict()
    merged.update(adapter)
    model.load_state_dict(merged, strict=True)


def assert_metrics_match(actual, expected, label):
    """Reproduce selection-time validation metrics before opening test scoring."""
    if actual['windows'] != expected['windows']:
        raise ValueError(f'{label}: validation window count changed')
    if len(actual['per_horizon']) != len(expected['per_horizon']):
        raise ValueError(f'{label}: forecast horizon changed')
    largest_difference = 0.0
    pairs = [(actual[name], expected[name]) for name in
             ('raw_masked', 'horizon_average', 'standardized_unmasked')]
    pairs.extend(zip(actual['per_horizon'], expected['per_horizon']))
    for observed, reference in pairs:
        if set(observed) != set(reference):
            raise ValueError(f'{label}: metric fields changed')
        for name in observed:
            a, b = observed[name], reference[name]
            if name in ('step', 'valid'):
                if a != b:
                    raise ValueError(f'{label}: {name} changed')
            elif not np.isfinite(a) or not np.isclose(a, b, rtol=1e-6, atol=1e-6):
                raise ValueError(f'{label}: {name} differs: restored={a}, frozen={b}')
            largest_difference = max(largest_difference, abs(float(a)-float(b)))
    return dict(status='PASS', windows=actual['windows'],
                largest_absolute_metric_difference=largest_difference,
                tolerance=dict(rtol=1e-6, atol=1e-6))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def restore_tqnet_kv(root, arrays, protocol, seed, device):
    base_path = root / f'runs/seed{seed}/baseline.pt'
    adapter_path = root / f'runs/seed{seed}/adapter.pt'
    base_payload = torch.load(base_path, map_location='cpu', weights_only=False)
    adapter_payload = torch.load(adapter_path, map_location='cpu', weights_only=False)
    validate_checkpoint_metadata(base_payload, protocol, seed, vars(config()))
    validate_checkpoint_metadata(adapter_payload, protocol, seed)
    if base_payload.get('version') != VERSION or adapter_payload.get('version') != VERSION:
        raise ValueError('checkpoint recipe version differs from the frozen recipe')
    base_hash = sha(base_path)
    if adapter_payload.get('base_checkpoint_sha256') != base_hash:
        raise ValueError('adapter is not paired with this exact frozen baseline')
    baseline_record = _read_json(root / f'results/baseline_validation_seed{seed}.json')
    paired_record = _read_json(root / f'results/paired_validation_seed{seed}.json')
    for record in (baseline_record, paired_record):
        validate_validation_record(record, protocol, seed)
    if baseline_record.get('checkpoint_sha256') != base_hash:
        raise ValueError('baseline validation/checkpoint pairing changed')
    if paired_record.get('base_checkpoint_sha256') != base_hash:
        raise ValueError('paired validation/checkpoint pairing changed')
    if paired_record.get('horizon_gate') != adapter_payload.get('horizon_gate'):
        raise ValueError('stored gate differs between checkpoint and validation record')
    weights = frozen_gate(adapter_payload['horizon_gate'])
    if paired_record.get('best_epoch') != adapter_payload['best_epoch']:
        raise ValueError('adapter epoch differs from validation selection')
    if baseline_record.get('best_epoch') != base_payload['best_epoch']:
        raise ValueError('baseline epoch differs from validation selection')
    graph = load_graph(root/'data/physical/sd_rn_adj.npy', root/'data/physical/sd_meta.csv',
                       expected_sensor_ids=arrays['node_ids'])
    # Paths legitimately change in a copied project; structure and source bytes do not.
    stable_graph_fields = ('version', 'edges_sha256', 'sensor_order_sha256',
                           'adjacency_source_sha256', 'metadata_source_sha256',
                           'selected_edges', 'covered_variables', 'max_edge_size', 'edge_budget')
    for field in stable_graph_fields:
        if graph['metadata'].get(field) != paired_record['physical_hypergraph'].get(field):
            raise ValueError(f'physical graph changed: {field}')
    cutoff = int(arrays['origins_val'][0])-LENGTH-1
    if paired_record.get('memory_train_only') is not True or paired_record.get('memory_template_fit_end') != cutoff:
        raise ValueError('memory/template training cutoff changed')
    daily, weekly = periodic_prior(arrays['data'][:cutoff], PERIOD, PERIOD*7)
    base = TQNet(config())
    base.load_state_dict(base_payload['model'], strict=True)
    model = TQNetPhysicsKV(base, graph, daily, weekly, HORIZON, 96)
    restore_adapter_state(model, adapter_payload['adapter'])
    torch.testing.assert_close(model.incidence, graph['incidence'], rtol=0, atol=0)
    torch.testing.assert_close(model.edge_pool, graph['edge_pool'], rtol=0, atol=0)
    torch.testing.assert_close(model.daily, daily, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model.weekly, weekly, rtol=1e-6, atol=1e-7)
    model.to(device).eval()
    # Restore templates first: memory keys/values must use the checkpoint buffers.
    bank, sampling = create_bank(arrays, graph, model.daily, model.weekly, device, False)
    if sampling != adapter_payload.get('sampling') or sampling != paired_record.get('sampling'):
        raise ValueError('memory origin sampling differs from the frozen run')
    if bank.normalization_metadata != adapter_payload.get('normalization') or bank.normalization_metadata != paired_record.get('normalization'):
        raise ValueError('memory feature normalization changed')
    if not np.all(bank.origins.cpu().numpy()+HORIZON <= cutoff):
        raise ValueError('memory contains non-training targets')
    scores = evaluate(model, arrays, arrays['origins_val'], device, bank, weights)
    verification = {
        'baseline': assert_metrics_match(scores['base'], baseline_record['metrics'], 'baseline'),
        'paired_base': assert_metrics_match(scores['base'], paired_record['metrics']['base'], 'paired baseline'),
        'paired_kv': assert_metrics_match(scores['kv'], paired_record['metrics']['kv'], 'paired KV'),
    }
    return model, bank, weights, dict(
        validation_restore=verification, base_checkpoint=str(base_path),
        base_checkpoint_sha256=base_hash, adapter_checkpoint=str(adapter_path),
        adapter_checkpoint_sha256=sha(adapter_path),
        best_epoch=dict(base=base_payload['best_epoch'], kv=adapter_payload['best_epoch']),
        horizon_gate=adapter_payload['horizon_gate'], sampling=sampling,
        normalization=bank.normalization_metadata, physical_hypergraph=graph['metadata'],
        memory_train_only=True, memory_template_fit_end=cutoff,
        paired_same_checkpoint=True, base_weights_unchanged=True,
    )


@torch.no_grad()
def evaluate_strong(model, arrays, origins, device):
    model.eval()
    scores = Metrics(HORIZON, arrays['mean'], arrays['std'])
    for x, y, absolute_origins in loader(arrays, origins):
        # Keep the absolute timestamp; taking %96 would destroy day-of-week.
        prediction = model(x.to(device), absolute_origins.to(device))
        scores.add(prediction, y.to(device))
    return scores.result()


def restore_strong(root, arrays, protocol, name, seed, device):
    from strong_baselines import build_model, model_cfg
    path = root / f'runs/{name}/seed{seed}/model.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_metadata(payload, protocol, seed, model_cfg(name))
    record = _read_json(root/f'results/{name}_validation_seed{seed}.json')
    validate_validation_record(record, protocol, seed)
    if record.get('best_epoch') != payload['best_epoch']:
        raise ValueError('strong baseline validation-selected epoch changed')
    if 'checkpoint_sha256' in record and record['checkpoint_sha256'] != sha(path):
        raise ValueError('strong baseline checkpoint differs from validation record')
    model = build_model(name, root, device)
    model.load_state_dict(payload['model'], strict=True)
    scores = evaluate_strong(model, arrays, arrays['origins_val'], device)
    verification = assert_metrics_match(scores, record['metrics'], name)
    return model, dict(checkpoint=str(path), checkpoint_sha256=sha(path),
                       best_epoch=payload['best_epoch'], configuration=payload['config'],
                       validation_restore=verification)


def origins_metadata(origins):
    origins = np.asarray(origins, dtype='<i8')
    used = origins[:len(origins)//EVAL_BATCH*EVAL_BATCH]
    if not len(used):
        raise ValueError('test contains no full official evaluation batches')
    return dict(available_windows=len(origins), windows=len(used),
                dropped_remainder=len(origins)-len(used), batch_size=EVAL_BATCH,
                drop_last=True, first_forecast_origin=int(used[0]),
                last_forecast_origin=int(used[-1]),
                evaluated_origins_sha256=hashlib.sha256(used.tobytes()).hexdigest())


def write_new_json(path, value):
    """Publish a complete JSON file without replacing an existing evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        # An exclusive hard link is atomic on the local project filesystem.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('only physical GPU2 is authorized for this comparison')
    if not torch.cuda.is_available():
        raise RuntimeError('authorized CUDA device is unavailable')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(args.root).resolve()
    if root != Path(__file__).resolve().parent:
        raise RuntimeError('root must be the isolated comparison project')
    filename = f'paired_test_seed{args.seed}.json' if args.model == 'tqnet_kv' else f'{args.model}_test_seed{args.seed}.json'
    output = root/'results'/filename
    if output.exists():
        raise FileExistsError(f'test result is immutable: {output}')
    validate_hashes(root)
    release_hash = sha(root/RELEASE)
    arrays, protocol, data_path = prepared(root, include_test=True)
    device = torch.device('cuda:0')
    if args.model == 'tqnet_kv':
        model, bank, weights, audit = restore_tqnet_kv(root, arrays, protocol, args.seed, device)
    else:
        model, audit = restore_strong(root, arrays, protocol, args.model, args.seed, device)
    # Every restoration/validation check completes before the first test score.
    validate_hashes(root)
    if sha(root/RELEASE) != release_hash:
        raise RuntimeError('test release changed during restoration')
    if args.model == 'tqnet_kv':
        scores = evaluate(model, arrays, arrays['origins_test'], device, bank, weights)
    else:
        scores = {args.model: evaluate_strong(model, arrays, arrays['origins_test'], device)}
    window_audit = origins_metadata(arrays['origins_test'])
    if window_audit['windows'] != 6976 or any(score['windows'] != 6976 for score in scores.values()):
        raise RuntimeError('official fixed test window count changed')
    validate_hashes(root)
    if sha(root/RELEASE) != release_hash:
        raise RuntimeError('test release changed during evaluation')
    result = dict(model=args.model, dataset='LargeST-SD', seed=args.seed,
                  split='test', seq_len=LENGTH, horizon=HORIZON, protocol=protocol,
                  primary_metric='horizon_average', metrics=scores,
                  window_audit=window_audit, release_sha256=release_hash,
                  prepared_data_sha256=sha(data_path), smoke=False, physical_gpu=2,
                  test_evaluated=True, test_labels_used_for_selection=False,
                  test_labels_used_for_memory=False, test_calibration_performed=False,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, **audit)
    write_new_json(output, result)
    print('LARGEST_LOCKED_TEST_RESULT '+json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
