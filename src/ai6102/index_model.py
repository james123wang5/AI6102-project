"""Synthetic-index weights fitted from training labels, and index-aware post-processing.

By definition target_i(t) = 1e4 * (r_i(t) - r_index(t)), where r is the
60-second forward WAP return. On training rows both r_i (from the WAP 60 s
later) and target are known, so the index return is implied as
r_i - target / 1e4. Static stock weights are then fitted by non-negative
least squares of r_index on the r_i, using training dates only. At
prediction time only the fitted weights and current WAPs are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ai6102.features.operators import ROW_KEYS, TIME_KEYS, exact_time_lag


N_STOCKS = 200


@dataclass(frozen=True)
class IndexFit:
    weights: np.ndarray          # shape (N_STOCKS,), non-negative, sums to 1
    r2: float                    # fit of implied index return on weighted stock returns
    n_times: int
    implied_dispersion: float    # median cross-stock std of the implied index return (bp)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"stock_id": np.arange(N_STOCKS), "weight": self.weights})


def _stock_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame["stock_id"].astype("int16").to_numpy()


def fit_index_weights(train: pd.DataFrame, target_column: str = "target") -> IndexFit:
    """Fit static index weights from training rows only (needs target and wap)."""
    data = train.loc[:, ROW_KEYS + ["wap", target_column]].copy()
    data["stock_id"] = _stock_ids(data)
    # exact_time_lag with a negative lag reads the WAP 60 seconds later.
    future_wap = exact_time_lag(data, "wap", -60)
    data["stock_return"] = (future_wap / data["wap"] - 1.0).astype("float64")
    data["implied"] = data["stock_return"] - data[target_column].astype("float64") / 1e4
    data = data.loc[np.isfinite(data["stock_return"]) & np.isfinite(data["implied"])]

    per_time = data.groupby(TIME_KEYS, sort=True)["implied"]
    y = per_time.median()
    dispersion = float(per_time.std().median() * 1e4)
    returns = data.pivot_table(index=TIME_KEYS, columns="stock_id", values="stock_return", aggfunc="first")
    returns = returns.reindex(index=y.index, columns=range(N_STOCKS)).fillna(0.0)

    matrix = returns.to_numpy()
    weights, _ = nnls(matrix, y.to_numpy(), maxiter=50 * N_STOCKS)
    fitted = matrix @ weights
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else float("nan")
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return IndexFit(weights=weights, r2=r2, n_times=len(y), implied_dispersion=dispersion)


def weighted_time_mean(frame: pd.DataFrame, values: pd.Series | np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Per-(date, second) weighted mean of `values` over the stocks present."""
    values = np.asarray(values, dtype="float64")
    w = np.ones(len(frame)) if weights is None else weights[_stock_ids(frame)]
    w = np.where(np.isfinite(values), w, 0.0)
    keys = [frame[k].to_numpy() for k in TIME_KEYS]
    helper = pd.DataFrame({"d": keys[0], "s": keys[1], "wv": w * np.nan_to_num(values), "w": w})
    grouped = helper.groupby(["d", "s"], sort=False)
    numerator = grouped["wv"].transform("sum").to_numpy()
    denominator = grouped["w"].transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def neutralize(frame: pd.DataFrame, prediction: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Subtract the per-time (weighted) mean prediction.

    Targets are returns relative to the weighted index, so their weighted
    cross-sectional mean is close to zero; predictions are made to satisfy it
    too. `weights=None` means equal weights (an ablation of the fitted ones).
    """
    mean = weighted_time_mean(frame, prediction, weights)
    return np.asarray(prediction, dtype="float64") - np.nan_to_num(mean)
