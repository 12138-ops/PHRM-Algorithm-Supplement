"""Shared, explicit PEMS08 splits for paired TQNet and KV experiments."""

from pathlib import Path

import numpy as np


def load_pems08(root: Path, horizon: int, split: str):
    with np.load(root / "dataset" / "PEMS08_data.npz", allow_pickle=False) as payload:
        raw_source = payload["data"][:, :, 0]
    n = len(raw_source)

    if split == "60/20/20":
        # Match the released TQNet Dataset_PEMS class: floor each ratio,
        # fit StandardScaler on train rows only, and allow history to cross
        # split boundaries while targets stay inside each split.
        from sklearn.preprocessing import StandardScaler

        train_end = int(n * 0.60)
        num_valid = int(n * 0.20)
        num_test = int(n * 0.20)
        val_end = train_end + num_valid
        test_start = n - num_test
        test_stop = n - horizon + 1
        scaler = StandardScaler().fit(raw_source[:train_end])
        data = scaler.transform(raw_source).astype(np.float32)
    else:
        raise ValueError(f"unsupported main-table split: {split}")

    train_origins = np.arange(96, train_end - horizon + 1, dtype=np.int64)
    val_origins = np.arange(train_end, val_end - horizon + 1, dtype=np.int64)
    test_origins = np.arange(test_start, test_stop, dtype=np.int64)
    if not all(len(x) for x in (train_origins, val_origins, test_origins)):
        raise ValueError(f"empty PEMS08 split for horizon={horizon}, split={split}")
    assert train_origins[-1] + horizon <= train_end
    assert val_origins[0] >= train_end and val_origins[-1] + horizon <= val_end
    assert test_origins[0] >= test_start and test_origins[-1] + horizon <= n
    return data, train_end, val_end, test_start, train_origins, val_origins, test_origins
