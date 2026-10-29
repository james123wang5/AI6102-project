"""Stage-04 batch-5 neural models on the same features, folds and protocol as LightGBM.

Rows are laid out as a dense tensor [day, time step (55), stock (200), feature].
Two views of that tensor:

- GRU (time axis): one sequence of 55 steps per (stock, day). Unidirectional,
  so the output at step t only reads features at steps <= t, which are
  themselves causal.
- Transformer (stock axis): one set of up to 200 stocks per (day, time step),
  no positional encoding, absent stocks masked. The output subtracts the
  index-weighted mean over the stocks present, so predictions satisfy the
  same zero-sum identity as the target (index weights fitted on the fold's
  training dates only).

Training mirrors the trees: L1 loss with recency weights, early stopping on
the last training days, then a refit on all training days for the selected
number of epochs (same learning-rate trajectory).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai6102.index_model import N_STOCKS

N_STEPS = 55


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- data


@dataclass
class Scaler:
    """Signed-log for heavy-tailed columns, then standardise and clip; fitted on training rows."""

    log_mask: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    clip: float

    @classmethod
    def fit(cls, values: np.ndarray, clip: float, seed: int = 0) -> "Scaler":
        rng = np.random.default_rng(seed)
        sample = values[rng.choice(len(values), size=min(len(values), 1_000_000), replace=False)].astype("float64")
        with np.errstate(all="ignore"):
            log_mask = np.nanpercentile(np.abs(sample), 99, axis=0) > 50
            sample = np.where(log_mask, np.sign(sample) * np.log1p(np.abs(sample)), sample)
            mean = np.nanmean(sample, axis=0)
            std = np.nanstd(sample, axis=0)
        mean = np.nan_to_num(mean)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return cls(log_mask, mean.astype("float32"), std.astype("float32"), clip)

    def transform(self, values: np.ndarray, chunk: int = 500_000) -> np.ndarray:
        out = np.empty(values.shape, dtype="float32")
        for start in range(0, len(values), chunk):
            x = values[start:start + chunk].astype("float32")
            with np.errstate(all="ignore"):
                x = np.where(self.log_mask, np.sign(x) * np.log1p(np.abs(x)), x)
                x = (x - self.mean) / self.std
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -self.clip, self.clip)
            out[start:start + chunk] = x
        return out


@dataclass
class Grid:
    """Dense day x step x stock layout of one row set."""

    days: np.ndarray            # sorted date_ids, shape (D,)
    x: np.ndarray               # (D, 55, 200, F) float32, 0 where absent
    y: np.ndarray               # (D, 55, 200) float32, 0 where unknown
    present: np.ndarray         # (D, 55, 200) bool: row exists (features usable)
    labelled: np.ndarray        # (D, 55, 200) bool: row exists and target finite
    index: tuple[np.ndarray, np.ndarray, np.ndarray]  # row -> (d, t, s) for the source frame


def to_grid(frame: pd.DataFrame, columns: list[str], scaler: Scaler, target_column: str = "target") -> Grid:
    days = np.unique(frame["date_id"].to_numpy())
    d = np.searchsorted(days, frame["date_id"].to_numpy())
    t = (frame["seconds_in_bucket"].to_numpy() // 10).astype("int64")
    s = frame["stock_id"].astype("int16").to_numpy().astype("int64")
    shape = (len(days), N_STEPS, N_STOCKS)
    x = np.zeros(shape + (len(columns),), dtype="float32")
    x[d, t, s] = scaler.transform(frame[columns].to_numpy())
    target = frame[target_column].to_numpy(dtype="float32")
    finite = np.isfinite(target)
    y = np.zeros(shape, dtype="float32")
    y[d, t, s] = np.where(finite, target, 0.0)
    present = np.zeros(shape, dtype=bool)
    present[d, t, s] = True
    labelled = np.zeros(shape, dtype=bool)
    labelled[d, t, s] = finite
    return Grid(days, x, y, present, labelled, (d, t, s))


# ---------------------------------------------------------------- models


def build_model(kind: str, n_features: int, cfg: dict):
    import torch
    from torch import nn

    hidden, dropout = int(cfg["hidden"]), float(cfg["dropout"])

    def stem() -> nn.Module:
        return nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout))

    def head() -> nn.Module:
        return nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    class GRUNet(nn.Module):
        """x (B, 55, S, F) -> prediction (B, 55, S); one causal sequence per stock-day."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = stem()
            self.gru = nn.GRU(hidden, hidden, int(cfg["gru_layers"]), batch_first=True, dropout=dropout)
            self.head = head()

        def forward(self, x, present, weights):
            b, t, s, f = x.shape
            seq = x.permute(0, 2, 1, 3).reshape(b * s, t, f)
            out, _ = self.gru(self.stem(seq))
            return self.head(out).reshape(b, s, t).permute(0, 2, 1)

    class StockTransformer(nn.Module):
        """x (B, 55, S, F) -> prediction (B, 55, S); attention across stocks at each time step."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = stem()
            layer = nn.TransformerEncoderLayer(hidden, int(cfg["heads"]), 4 * hidden, dropout,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, int(cfg["tfm_layers"]), enable_nested_tensor=False)
            self.head = head()

        def forward(self, x, present, weights):
            b, t, s, f = x.shape
            tokens = x.reshape(b * t, s, f)
            mask = present.reshape(b * t, s)
            empty = ~mask.any(dim=1)
            pad = ~mask
            pad[empty] = False  # avoid all-masked rows (NaN); their outputs are never used
            out = self.head(self.encoder(self.stem(tokens), src_key_padding_mask=pad)).squeeze(-1)
            w = weights.unsqueeze(0) * mask
            mean = (out * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return (out - mean).reshape(b, t, s)

    if kind == "gru":
        return GRUNet()
    if kind == "tfm":
        return StockTransformer()
    raise ValueError(kind)


# ---------------------------------------------------------------- training


class Trainer:
    def __init__(self, kind: str, cfg: dict, device: str, index_weights: np.ndarray, seed: int, max_epochs: int) -> None:
        import torch

        self.torch = torch
        self.kind, self.cfg, self.device, self.seed = kind, cfg, device, seed
        self.max_epochs = max_epochs
        self.weights = torch.as_tensor(index_weights, dtype=torch.float32, device=device)
        self.batch_days = int(cfg[f"days_per_batch_{kind}"])
        self.amp = device == "cuda"

    def _tensors(self, grid: Grid, day_weights: np.ndarray | None):
        torch = self.torch
        dw = np.ones(len(grid.days), dtype="float32") if day_weights is None else day_weights.astype("float32")
        return {
            "x": torch.as_tensor(grid.x, device=self.device),
            "y": torch.as_tensor(grid.y, device=self.device),
            "present": torch.as_tensor(grid.present, device=self.device),
            "labelled": torch.as_tensor(grid.labelled, device=self.device),
            "dw": torch.as_tensor(dw, device=self.device),
        }

    def _forward(self, model, data, idx):
        with self.torch.autocast("cuda", dtype=self.torch.bfloat16, enabled=self.amp):
            return model(data["x"][idx], data["present"][idx], self.weights).float()

    def _lr(self, step: int, steps_per_epoch: int) -> float:
        base = float(self.cfg["lr"])
        warmup = int(self.cfg["warmup_epochs"]) * steps_per_epoch
        total = self.max_epochs * steps_per_epoch
        if step < warmup:
            return base * (step + 1) / warmup
        return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    def fit(self, train: Grid, day_weights: np.ndarray | None, valid: Grid | None, epochs: int, tag: str):
        """Train for up to `epochs`; with `valid`, early-stop and return (model, best_epoch)."""
        torch = self.torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        model = build_model(self.kind, train.x.shape[-1], self.cfg).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.cfg["lr"]),
                                      weight_decay=float(self.cfg["weight_decay"]))
        data = self._tensors(train, day_weights)
        n_days = len(train.days)
        steps_per_epoch = math.ceil(n_days / self.batch_days)
        best, best_epoch, best_state, waited, step = math.inf, 0, None, 0, 0
        patience = int(self.cfg["patience"])
        for epoch in range(1, epochs + 1):
            model.train()
            started, total, count = time.perf_counter(), 0.0, 0.0
            for batch in np.array_split(rng.permutation(n_days), steps_per_epoch):
                for group in optimizer.param_groups:
                    group["lr"] = self._lr(step, steps_per_epoch)
                idx = torch.as_tensor(batch, device=self.device)
                pred = self._forward(model, data, idx)
                mask = data["labelled"][idx]
                w = data["dw"][idx].view(-1, 1, 1) * mask
                loss = ((pred - data["y"][idx]).abs() * w).sum() / w.sum().clamp_min(1e-12)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.cfg["grad_clip"]))
                optimizer.step()
                step += 1
                total += float(loss.detach()) * float(mask.sum())
                count += float(mask.sum())
            message = f"    {tag} epoch {epoch}: train L1 {total / max(count, 1):.5f}"
            if valid is not None:
                score = mae_on_grid(self.predict(model, valid), valid)
                message += f"  inner MAE {score:.5f}"
                if score < best - 1e-5:
                    best, best_epoch, waited = score, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    waited += 1
            log(message + f"  ({time.perf_counter() - started:.0f}s)")
            if valid is not None and waited >= patience:
                break
        del data
        if valid is not None and best_state is not None:
            model.load_state_dict(best_state)
        return model, (best_epoch if valid is not None else epochs)

    def predict(self, model, grid: Grid) -> np.ndarray:
        torch = self.torch
        model.eval()
        data = self._tensors(grid, None)
        out = np.zeros(grid.y.shape, dtype="float32")
        with torch.no_grad():
            for start in range(0, len(grid.days), self.batch_days):
                idx = torch.arange(start, min(start + self.batch_days, len(grid.days)), device=self.device)
                out[start:start + len(idx)] = self._forward(model, data, idx).cpu().numpy()
        del data
        return out


def mae_on_grid(pred: np.ndarray, grid: Grid) -> float:
    return float(np.abs(pred - grid.y)[grid.labelled].mean())
