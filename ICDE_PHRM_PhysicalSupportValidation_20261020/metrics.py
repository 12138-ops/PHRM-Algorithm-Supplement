"""Streaming raw-unit LargeST metrics; zero targets are missing observations."""
import numpy as np
import torch


def masked_mae_loss(prediction, raw_target, mean, std):
    mask = raw_target.ne(0)
    error = (prediction * std + mean - raw_target).abs()
    return (error * mask).sum() / mask.sum().clamp_min(1)


class Metrics:
    def __init__(self, horizon, mean, std):
        self.mean, self.std = float(mean), float(std)
        self.totals = np.zeros((horizon, 4), dtype=np.float64)
        self.standardized = np.zeros(3, dtype=np.float64)
        self.windows = 0

    @torch.no_grad()
    def add(self, prediction, raw_target):
        raw_pred = prediction.double() * self.std + self.mean
        truth = raw_target.double()
        mask = truth.ne(0)
        error = raw_pred - truth
        vals = torch.stack([
            (error.abs() * mask).sum((0, 2)),
            (error.square() * mask).sum((0, 2)),
            (error.abs() / truth.clamp_min(1) * mask).sum((0, 2)),
            mask.sum((0, 2)),
        ], dim=-1)
        self.totals += vals.cpu().numpy()
        normalized_error = error / self.std
        self.standardized += [float(normalized_error.square().sum()),
                              float(normalized_error.abs().sum()), truth.numel()]
        self.windows += truth.shape[0]

    def result(self):
        if self.windows == 0 or np.any(self.totals[:, 3] == 0):
            raise ValueError('no observed validation targets')
        ae, se, ape, count = self.totals.T
        rows = [dict(step=i+1, mae=float(ae[i]/count[i]),
                     rmse=float(np.sqrt(se[i]/count[i])),
                     mse=float(se[i]/count[i]), mape=float(ape[i]/count[i]),
                     mape_percent=float(100*ape[i]/count[i]), valid=int(count[i]))
                for i in range(len(ae))]
        return dict(
            windows=self.windows, raw_masked=dict(
                mae=float(ae.sum()/count.sum()), mse=float(se.sum()/count.sum()),
                rmse=float(np.sqrt(se.sum()/count.sum())),
                mape=float(ape.sum()/count.sum()),
                mape_percent=float(100*ape.sum()/count.sum())),
            per_horizon=rows,
            horizon_average={k:float(np.mean([r[k] for r in rows]))
                             for k in ('mae','mse','rmse','mape','mape_percent')},
            standardized_unmasked=dict(mse=float(self.standardized[0]/self.standardized[2]),
                                       mae=float(self.standardized[1]/self.standardized[2])))
