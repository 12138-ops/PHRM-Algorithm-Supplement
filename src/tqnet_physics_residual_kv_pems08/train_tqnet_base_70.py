"""Train a paired TQNet baseline under the same chronological 70/10/20 split."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from models.tqnet import Model as TQNet
from pems08_protocol import load_pems08
from train_tqnet_physicskv import loader, make_config, evaluate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_base(model, data_loader, device):
    model.eval()
    mse_sum = mae_sum = 0.0
    count = 0
    for x, y, origins in data_loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        cycle = origins.to(device=device, dtype=torch.long).remainder(model.cycle_len)
        pred = model(x, cycle)
        err = pred - y
        mse_sum += float(err.square().sum(dtype=torch.float64))
        mae_sum += float(err.abs().sum(dtype=torch.float64))
        count += y.numel()
    return {"mse": mse_sum / max(count, 1), "mae": mae_sum / max(count, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--horizon", type=int, required=True, choices=[12, 24, 48, 96])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split", choices=["70/10/20", "60/20/20"], default="70/10/20")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_root", default="base_checkpoints_70")
    a = p.parse_args()
    seed_everything(a.seed)
    root = Path(a.root).resolve()
    h = a.horizon
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    data, train_end, val_end, test_start, train_origins, val_origins, test_origins = load_pems08(root, h, a.split)
    train_loader = loader(data, train_origins, h, a.batch_size, a.workers, True)
    val_loader = loader(data, val_origins, h, a.batch_size, a.workers, False)
    test_loader = loader(data, test_origins, h, a.batch_size, a.workers, False)
    model = TQNet(make_config(h)).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, a.epochs + 1):
        # TQNet's released PEMS recipe uses lradj=type3: keep the initial
        # learning rate for the first three epochs, then decay by 0.8 each epoch.
        lr = a.lr if epoch <= 3 else a.lr * (0.8 ** (epoch - 3))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = steps = 0.0
        started = time.monotonic()
        for x, y, origins in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            cycle = origins.to(device=device, dtype=torch.long).remainder(model.cycle_len)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, cycle)
            loss = (pred - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        val = eval_base(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": total / max(steps, 1), "val": val})
        print(
            f"epoch={epoch} train_mse={total/max(steps,1):.6f} "
            f"val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} lr={lr:.7g} "
            f"seconds={time.monotonic()-started:.1f}",
            flush=True,
        )
        if val["mse"] < best:
            best = val["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= a.patience:
                print("early_stop", flush=True)
                break
    if best_state is None:
        raise RuntimeError("no paired TQNet checkpoint was produced")
    model.load_state_dict(best_state)
    val = eval_base(model, val_loader, device)
    test = eval_base(model, test_loader, device)
    out = root / a.output_root / f"h{h}" / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "checkpoint.pth")
    result = {
        "model": "TQNet",
        "horizon": h,
        "seed": a.seed,
        "split": a.split,
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
        "best_epoch": best_epoch,
        "validation": val,
        "test": test,
        "test_labels_used_for_selection": False,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("TQNET_BASE_70_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
