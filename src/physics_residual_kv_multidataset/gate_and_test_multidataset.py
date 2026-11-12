"""Validation-only horizon gate for the 60/20/20 PEMS experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backbone import physical_candidates
from train_kv_multidataset import batch, dl, load_model, make_bank


GRID = tuple(float(x) for x in np.linspace(0.0, 1.0, 9))


@torch.no_grad()
def calibrate(model, loader, bank, mean, std, device):
    model.eval()
    horizon = int(model.horizon)
    mse = torch.zeros(horizon, len(GRID), dtype=torch.float64)
    mae = torch.zeros(horizon, len(GRID), dtype=torch.float64)
    count = 0
    for item in loader:
        x, y, origins = batch(item, device)
        pred, cache = model(x, origins, bank)
        base = cache["base"]
        residual = pred - base
        for j, weight in enumerate(GRID):
            error = (base + weight * residual - y) / std
            mse[:, j] += error.square().sum(dim=(0, 2)).cpu().to(torch.float64)
            mae[:, j] += error.abs().sum(dim=(0, 2)).cpu().to(torch.float64)
        count += int(y.shape[0] * y.shape[2])
    if count == 0:
        raise RuntimeError("empty validation loader")
    mse = mse / count
    mae = mae / count
    best = torch.argmin(mse, dim=1)
    weights = [GRID[int(index)] for index in best.tolist()]
    return {
        "grid": list(GRID),
        "weights": weights,
        "best_index": best.tolist(),
        "validation_step_mse": mse[torch.arange(horizon), best].tolist(),
        "validation_step_mae": mae[torch.arange(horizon), best].tolist(),
        "validation_mse": float(mse[torch.arange(horizon), best].mean()),
        "validation_mae": float(mae[torch.arange(horizon), best].mean()),
        "validation_base_mse": mse[:, 0].tolist(),
        "validation_raw_mse": mse[:, -1].tolist(),
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
    }


@torch.no_grad()
def evaluate(model, loader, bank, mean, std, device, weights):
    model.eval()
    totals = {name: [0.0, 0.0] for name in ("base", "raw_trained", "trained")}
    count = 0
    gate = torch.tensor(weights, dtype=torch.float32, device=device)[None, :, None]
    for item in loader:
        x, y, origins = batch(item, device)
        raw, cache = model(x, origins, bank)
        base = cache["base"]
        selected = base + gate * (raw - base)
        for name, value in (("base", base), ("raw_trained", raw), ("trained", selected)):
            error = (value - y) / std
            totals[name][0] += float(error.square().sum(dtype=torch.float64))
            totals[name][1] += float(error.abs().sum(dtype=torch.float64))
        count += y.numel()
    return {
        name: {"mse": mse / max(count, 1), "mae": mae / max(count, 1)}
        for name, (mse, mae) in totals.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--edge_csv", required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory_stride", type=int, default=12)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--edge_budget", type=int, default=512)
    parser.add_argument("--max_edge_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset = args.dataset.upper()
    horizon = int(args.horizon)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with np.load(args.data_path, allow_pickle=False) as payload:
        raw = payload["data"][:, :, 0].astype(np.float32)
    train_end = int(len(raw) * 0.6)
    val_end = train_end + int(len(raw) * 0.2)
    mean_np = raw[:train_end].mean(0).astype(np.float32)
    std_np = np.maximum(raw[:train_end].std(0), 1e-3).astype(np.float32)
    graph = physical_candidates(Path(args.edge_csv), args.num_nodes, args.edge_budget, args.max_edge_size)
    model, mean, std, daily, weekly = load_model(
        root, raw, train_end, graph, args, device, horizon, dataset
    )
    checkpoint = root / "runs" / dataset / f"h{horizon}" / f"seed{args.seed}" / "physical.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    bank = make_bank(raw, train_end, graph, args, device, horizon, mean, std, daily, weekly)
    val_loader = dl(raw, np.arange(train_end, val_end - horizon + 1), args.batch_size, args.workers, False, horizon)
    test_loader = dl(raw, np.arange(val_end, len(raw) - horizon + 1), args.batch_size, args.workers, False, horizon)
    calibration = calibrate(model, val_loader, bank, mean, std, device)
    metrics = evaluate(model, test_loader, bank, mean, std, device, calibration["weights"])
    result = {
        "model": "PhysicsResidualKV_HorizonGate",
        "dataset": dataset,
        "horizon": horizon,
        "split": "test",
        "seed": args.seed,
        "protocol": "60/20/20",
        "train_end": train_end,
        "val_end": val_end,
        "memory_train_only": True,
        "test_labels_used_for_selection": False,
        "memory_episodes": int(bank.origins.numel()),
        "physical_edges": len(graph["edges"]),
        "horizon_gate": calibration,
        "metrics": metrics,
        "checkpoint": str(checkpoint),
    }
    out = root / "results" / dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / f"test_h{horizon}_seed{args.seed}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("PHYSICS_RESIDUAL_KV_GATE_TEST " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
