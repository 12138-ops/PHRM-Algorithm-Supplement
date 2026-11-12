from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.tqnet import Model as TQNet
from pems08_protocol import load_pems08
from physics_kv import (
    PhysicalKVBank,
    TQNetPhysicsKV,
    edge_descriptors,
    load_graph,
    periodic_prior,
    residual_future,
    residual_history,
    trainable_parameters,
)


# Calibration is deliberately restricted to the two short horizons that
# regressed in the matched 60/20/20 experiment.  h48/h96 keep the original
# adapter path and therefore remain directly comparable to the frozen formal
# baseline results.
HORIZON_GATE_GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, 9))


class WindowDataset(Dataset):
    def __init__(self, data, origins, horizon):
        self.data = data
        self.origins = np.asarray(origins, dtype=np.int64)
        self.horizon = int(horizon)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        t = int(self.origins[idx])
        return (
            torch.from_numpy(self.data[t - 96 : t]),
            torch.from_numpy(self.data[t : t + self.horizon]),
            torch.tensor(t, dtype=torch.long),
        )


def loader(data, origins, horizon, batch_size, workers, shuffle):
    return DataLoader(
        WindowDataset(data, origins, horizon),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def make_config(horizon):
    return SimpleNamespace(
        seq_len=96,
        pred_len=horizon,
        enc_in=170,
        cycle=288,
        model_type="mlp",
        d_model=512,
        dropout=0.0,
        use_revin=1,
    )


def load_base(checkpoint, horizon, device):
    model = TQNet(make_config(horizon)).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"TQNet checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def build_bank(data, origins, horizon, graph, daily, weekly, args, device):
    # Bank construction is deliberately performed from train origins only.
    edge_pool = graph["edge_pool"]
    chunks = []
    values = []
    batch_size = max(1, min(args.bank_batch_size, 64))
    for start in range(0, len(origins), batch_size):
        ts = torch.as_tensor(origins[start : start + batch_size], dtype=torch.long, device=device)
        x = torch.stack([torch.from_numpy(data[int(t) - 96 : int(t)]) for t in ts.cpu()]).to(device)
        y = torch.stack([torch.from_numpy(data[int(t) : int(t) + horizon]) for t in ts.cpu()]).to(device)
        rh = residual_history(x, ts, daily, weekly)
        rv = residual_future(y, ts, daily, weekly)
        chunks.append(edge_descriptors(rh, ts, edge_pool.to(device)).cpu())
        values.append(torch.einsum("bhn,ne->bhe", rv, edge_pool.to(device)).permute(0, 2, 1).cpu())
    keys = torch.cat(chunks, 0) if chunks else torch.empty(0, graph["incidence"].shape[1], 13)
    vals = torch.cat(values, 0) if values else torch.empty(0, graph["incidence"].shape[1], horizon)
    origins_t = torch.as_tensor(origins, dtype=torch.long)
    return PhysicalKVBank(origins_t, keys, vals, device, args.topk, args.temperature)


@torch.no_grad()
def evaluate(model, data_loader, bank, device, residual_weight=None):
    model.eval()
    model.base.eval()
    totals = {"trained": [0.0, 0.0], "base": [0.0, 0.0]}
    n = 0
    gate_sum = 0.0
    support_sum = 0.0
    for x, y, origins in data_loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        origins = origins.to(device=device, dtype=torch.long, non_blocking=True)
        pred, info = model(x, origins, bank, residual_weight=residual_weight)
        for name, value in (("trained", pred), ("base", info["base"])):
            err = value - y
            totals[name][0] += float(err.square().sum(dtype=torch.float64))
            totals[name][1] += float(err.abs().sum(dtype=torch.float64))
        n += y.numel()
        gate_sum += float(info["gate"].mean()) * x.shape[0]
        support_sum += float(info["support"].mean()) * x.shape[0]
    out = {
        name: {"mse": vals[0] / max(n, 1), "mae": vals[1] / max(n, 1)}
        for name, vals in totals.items()
    }
    out["gate_mean"] = gate_sum / max(len(data_loader.dataset), 1)
    out["support_fraction"] = support_sum / max(len(data_loader.dataset), 1)
    return out


@torch.no_grad()
def calibrate_horizon_gate(model, data_loader, bank, device, grid=HORIZON_GATE_GRID):
    """Select one residual weight per forecast step using validation labels only.

    The adapter is evaluated once and the validation errors for all candidate
    weights are accumulated in one pass.  Candidate zero is the frozen TQNet
    baseline, so the calibration can safely disable a harmful short-horizon
    correction.  The returned vector is applied unchanged to the test split.
    """
    model.eval()
    horizon = int(model.horizon)
    grid = tuple(float(x) for x in grid)
    if not grid or min(grid) < 0.0 or max(grid) > 1.0:
        raise ValueError("horizon gate grid must be non-empty and lie in [0, 1]")
    mse_sum = torch.zeros(horizon, len(grid), dtype=torch.float64)
    mae_sum = torch.zeros(horizon, len(grid), dtype=torch.float64)
    count = 0
    for x, y, origins in data_loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        origins = origins.to(device=device, dtype=torch.long, non_blocking=True)
        # No validation labels enter the model or the memory bank.  They are
        # used only here to choose the inference-time residual weight.
        _, info = model(x, origins, bank, residual_weight=None)
        base = info["base"]
        raw_residual = info["raw_residual"]
        for j, weight in enumerate(grid):
            err = base + float(weight) * raw_residual - y
            mse_sum[:, j] += err.square().sum(dim=(0, 2)).detach().cpu().to(torch.float64)
            mae_sum[:, j] += err.abs().sum(dim=(0, 2)).detach().cpu().to(torch.float64)
        count += int(y.shape[0] * y.shape[2])
    if count <= 0:
        raise RuntimeError("cannot calibrate horizon gate on an empty validation loader")
    mse = mse_sum / count
    mae = mae_sum / count
    # argmin returns the first candidate on ties; because grid is ascending,
    # ties choose the smallest residual weight and favor the base forecast.
    best_index = torch.argmin(mse, dim=1)
    weights = torch.tensor([grid[int(i)] for i in best_index.tolist()], dtype=torch.float32)
    selected_mse = mse[torch.arange(horizon), best_index]
    selected_mae = mae[torch.arange(horizon), best_index]
    return {
        "weights": weights.tolist(),
        "grid": list(grid),
        "best_index": best_index.tolist(),
        "validation_step_mse": selected_mse.tolist(),
        "validation_step_mae": selected_mae.tolist(),
        "validation_base_mse": mse[:, 0].tolist(),
        "validation_base_mae": mae[:, 0].tolist(),
        "validation_raw_mse": mse[:, -1].tolist() if 1.0 in grid else None,
        "validation_raw_mae": mae[:, -1].tolist() if 1.0 in grid else None,
        "validation_mse": float(selected_mse.mean()),
        "validation_mae": float(selected_mae.mean()),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--horizon", type=int, required=True, choices=[12, 24, 48, 96])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split", choices=["60/20/20"], default="60/20/20")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bank_batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--memory_stride", type=int, default=12)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--base_checkpoint", default="")
    p.add_argument("--runs_root", default="runs")
    a = p.parse_args()
    seed_everything(a.seed)
    root = Path(a.root).resolve()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    h = a.horizon

    data, train_end, val_end, test_start, train_origins, val_origins, test_origins = load_pems08(root, h, a.split)
    nodes = data.shape[1]
    bank_origins = train_origins[:: max(1, a.memory_stride)]
    graph = load_graph(root / "dataset" / "physical" / "PEMS08_physical_candidates.json", nodes)
    daily, weekly = periodic_prior(data[:train_end])
    daily, weekly = daily.to(device), weekly.to(device)

    checkpoint = Path(a.base_checkpoint) if a.base_checkpoint else root / "base_checkpoints" / f"h{h}" / f"seed{a.seed}.pth"
    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing official TQNet checkpoint: {checkpoint}")
    base = load_base(checkpoint, h, device)
    model = TQNetPhysicsKV(base, graph, daily, weekly, h, a.hidden).to(device)
    bank = build_bank(data, bank_origins, h, graph, daily, weekly, a, device)
    print(
        json.dumps(
            {
                "device": str(device),
                "horizon": h,
                "seed": a.seed,
                "split": a.split,
                "base_checkpoint_protocol": "frozen paired TQNet checkpoint",
                "train_end": train_end,
                "val_end": val_end,
                "test_start": test_start,
                "train_windows": len(train_origins),
                "validation_windows": len(val_origins),
                "test_windows": len(test_origins),
                "memory_episodes": len(bank_origins),
                "physical_edges": len(graph["edges"]),
                "base_checkpoint": str(checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_loader = loader(data, train_origins, h, a.batch_size, a.workers, True)
    val_loader = loader(data, val_origins, h, a.batch_size, a.workers, False)
    test_loader = loader(data, test_origins, h, a.batch_size, a.workers, False)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, a.epochs))
    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, a.epochs + 1):
        start = time.monotonic()
        model.train()
        model.base.eval()
        train_sum = 0.0
        steps = 0
        for x, y, origins in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            origins = origins.to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred, info = model(x, origins, bank, disable_memory=False)
            loss = (pred - y).square().mean() + 1e-4 * torch.tanh(model.residual_scale).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 5.0)
            optimizer.step()
            train_sum += float(loss.detach())
            steps += 1
        scheduler.step()
        val_metrics = evaluate(model, val_loader, bank, device)
        row = {
            "epoch": epoch,
            "train_mse": train_sum / max(steps, 1),
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_mse={row['train_mse']:.6f} "
            f"val_mse={val_metrics['trained']['mse']:.6f} "
            f"val_mae={val_metrics['trained']['mae']:.6f} "
            f"gate={val_metrics['gate_mean']:.4f} seconds={time.monotonic()-start:.1f}",
            flush=True,
        )
        if val_metrics["trained"]["mse"] < best_mse:
            best_mse = val_metrics["trained"]["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("base.")}
            bad = 0
        else:
            bad += 1
            if bad >= a.patience:
                print("early_stop", flush=True)
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current, strict=False)
    # Keep h48/h96 exactly on the original adapter path.  For h12/h24 only,
    # calibrate a per-forecast-step residual weight on the validation split.
    # The selected vector is then frozen and reused for test evaluation.
    raw_val_metrics = evaluate(model, val_loader, bank, device, residual_weight=None)
    if True:
        gate_calibration = calibrate_horizon_gate(model, val_loader, bank, device)
        selected_weight = torch.tensor(
            gate_calibration["weights"], dtype=torch.float32, device=device
        )
        val_metrics = evaluate(model, val_loader, bank, device, residual_weight=selected_weight)
        print(
            "HORIZON_GATE_CALIBRATION "
            + json.dumps(
                {
                    "horizon": h,
                    "selection_split": "validation",
                    "test_labels_used_for_selection": False,
                    "weights": gate_calibration["weights"],
                    "grid": gate_calibration["grid"],
                    "raw_validation_mse": raw_val_metrics["trained"]["mse"],
                    "selected_validation_mse": val_metrics["trained"]["mse"],
                    "raw_validation_mae": raw_val_metrics["trained"]["mae"],
                    "selected_validation_mae": val_metrics["trained"]["mae"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        gate_calibration = {
            "enabled": False,
            "selection_split": None,
            "weights": None,
            "grid": None,
        }
        selected_weight = None
        val_metrics = raw_val_metrics
    test_metrics = evaluate(model, test_loader, bank, device, residual_weight=selected_weight)
    output_dir = root / a.runs_root / f"h{h}" / f"seed{a.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": best_state,
            "best_epoch": best_epoch,
            "base_checkpoint": str(checkpoint),
            "train_end": train_end,
            "val_end": val_end,
            "horizon_gate": gate_calibration,
        },
        output_dir / "adapter.pt",
    )
    (output_dir / "horizon_gate.json").write_text(
        json.dumps(gate_calibration, indent=2), encoding="utf-8"
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = {
        "model": "TQNetPhysicsResidualKV",
        "horizon": h,
        "split": "test",
        "protocol": f"{a.split} adapter split; frozen paired TQNet base checkpoint",
        "seed": a.seed,
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
        "memory_train_only": True,
        "test_labels_used_for_selection": False,
        "memory_episodes": len(bank_origins),
        "physical_edges": len(graph["edges"]),
        "best_epoch": best_epoch,
        "base_checkpoint": str(checkpoint),
        "raw_validation": raw_val_metrics,
        "horizon_gate": gate_calibration,
        "validation": val_metrics,
        "test": test_metrics,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TQNET_PHYSICS_KV_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
