"""Pinned official STID/STGformer architectures in the shared LargeST protocol.

Forward origins are absolute FIRST-TARGET row indices, never clock phases.
Vendor sources remain byte-for-byte original.  Only STGformer's Mlp import is
redirected at load time to the separately pinned official timm Mlp dependency.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def model_cfg(name):
    """Return the immutable-source SD model/training settings, as plain JSON."""
    key = name.lower()
    if key not in ('stid', 'stgformer'):
        raise ValueError(f'Unsupported baseline {name!r}; use stid or stgformer')
    return json.loads((HERE / 'configs' / f'{key}_sd.json').read_text(encoding='utf-8'))


def verify_vendor_sources():
    provenance = json.loads((HERE / 'configs' / 'baseline_sources.json').read_text(encoding='utf-8'))
    for source in provenance['sources']:
        for record in source['files']:
            path = HERE / record['path']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record['sha256']:
                raise RuntimeError(f'Pinned official baseline source was modified: {path}')
    return provenance


def calendar_from_timestamps(timestamps_ns):
    """Exact clock fields from actual timestamps, including the weekday offset."""
    timestamps = np.asarray(timestamps_ns, dtype='datetime64[ns]')
    if timestamps.ndim != 1 or len(timestamps) < 24:
        raise ValueError('Expected at least 24 chronological timestamps')
    if not np.all(np.diff(timestamps).astype('timedelta64[m]') == np.timedelta64(15, 'm')):
        raise ValueError('The matched LargeST SD protocol requires regular 15-minute timestamps')
    dates = timestamps.astype('datetime64[D]')
    tod = ((timestamps - dates) / np.timedelta64(1, 'D')).astype(np.float32)
    # 1970-01-01 was Thursday, index 3 with Monday=0; no guessed dataset offset.
    dow = ((dates.astype(np.int64) + 3) % 7).astype(np.float32)
    return np.stack((tod, dow), axis=-1)


def _official_stgformer_type():
    source = HERE / 'vendor_baselines' / 'stgformer_official' / 'model' / 'STGformer.py'
    original = source.read_text(encoding='utf-8')
    old_import = 'from timm.models.vision_transformer import Mlp'
    new_import = 'from vendor_baselines.timm_official.timm.models.layers.mlp import Mlp'
    if original.count(old_import) != 1:
        raise RuntimeError('Pinned STGformer Mlp import no longer matches the audited source')
    module = types.ModuleType('_pinned_official_stgformer')
    module.__file__ = str(source)
    exec(compile(original.replace(old_import, new_import), str(source), 'exec'), module.__dict__)
    return module.STGformer


class OfficialBaseline(nn.Module):
    """Tensor-only feature adapter; no target or future values are accepted."""
    def __init__(self, name, timestamps_ns, *, num_nodes=716, support=None, model_args=None):
        super().__init__()
        self.name = name.lower()
        cfg = model_cfg(self.name)
        kwargs = copy.deepcopy(cfg['model_args'])
        kwargs['num_nodes'] = int(num_nodes)
        if model_args:
            kwargs.update(model_args)
        self.model_config = kwargs
        self.num_nodes = int(num_nodes)
        self.seq_len = 12
        self.register_buffer('calendar', torch.from_numpy(calendar_from_timestamps(timestamps_ns)), persistent=False)
        self.register_buffer('history_offsets', torch.arange(-self.seq_len, 0), persistent=False)
        if self.name == 'stid':
            official = importlib.import_module('vendor_baselines.stid_official.stid.arch').STID
            self.network = official(**kwargs)
        elif self.name == 'stgformer':
            if support is None or tuple(support.shape) != (num_nodes, num_nodes):
                raise ValueError('STGformer construction requires the ordered official SD support matrix')
            self.register_buffer('official_support', torch.as_tensor(support, dtype=torch.float32), persistent=False)
            self.network = _official_stgformer_type()(supports=[self.official_support], **kwargs)
        else:
            raise ValueError(self.name)

    def history_features(self, x, origins):
        if x.ndim != 3 or tuple(x.shape[1:]) != (12, self.num_nodes):
            raise ValueError(f'Expected [batch,12,{self.num_nodes}], got {tuple(x.shape)}')
        origins = torch.as_tensor(origins, dtype=torch.long, device=x.device)
        if origins.ndim != 1 or origins.shape[0] != x.shape[0]:
            raise ValueError('Pass one absolute first-target row index per history window')
        if bool((origins < 12).any()) or bool((origins >= len(self.calendar)).any()):
            raise ValueError('Absolute first-target row index outside prepared timestamp range')
        indices = origins[:, None] + self.history_offsets[None, :]
        fields = self.calendar[indices].to(dtype=x.dtype)
        if self.name == 'stid':
            fields = torch.stack((fields[..., 0], fields[..., 1] / 7), dim=-1)
        expanded = fields[:, :, None, :].expand(-1, -1, self.num_nodes, -1)
        return torch.cat((x[..., None], expanded), dim=-1)

    def forward(self, x, origins):
        history = self.history_features(x, origins)
        if self.name == 'stid':
            prediction = self.network(history, future_data=None, batch_seen=0, epoch=0, train=self.training)
        else:
            prediction = self.network(history)
        if prediction.shape != (x.shape[0], 12, self.num_nodes, 1):
            raise RuntimeError(f'Unexpected official {self.name} output shape {tuple(prediction.shape)}')
        return prediction[..., 0]


def build_model(name, root, device):
    """Use shared prepared data and return a trainable normalized forecaster."""
    verify_vendor_sources()
    root = Path(root)
    prepared = root / 'data' / 'processed' / 'LargeSTSD_2019_15min_l12_h12.npz'
    with np.load(prepared, allow_pickle=False) as arrays:
        timestamps = arrays['timestamps_ns'].copy()
        num_nodes = int(len(arrays['node_ids']))
    if num_nodes != 716:
        raise ValueError(f'Production LargeST-SD expects 716 ordered nodes, found {num_nodes}')
    support = None
    if name.lower() == 'stgformer':
        adjacency = np.load(root / 'data' / 'physical' / 'sd_rn_adj.npy', allow_pickle=False)
        from vendor_baselines.stgformer_official.lib.graph_algo import normalize_adj_mx
        support = np.asarray(normalize_adj_mx(adjacency, 'normlap')[0], dtype=np.float32)
    return OfficialBaseline(name, timestamps, num_nodes=num_nodes, support=support).to(device)
