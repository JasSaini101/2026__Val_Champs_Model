"""Scoring probabilistic predictions: overall, on cross-region maps and on internationals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6
METRICS = ("log_loss", "brier", "accuracy", "ece")
SEGMENTS = {
    "overall": None,
    "cross_region": "cross_region",
    "international": "is_international",
}


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p > 0.5) == (y == 1)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average |predicted - observed| win rate over probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(y)) if len(y) else float("nan")


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return {"n": 0, **{m: float("nan") for m in METRICS}}
    return {
        "n": len(y),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "accuracy": accuracy(y, p),
        "ece": ece(y, p),
    }


def evaluate(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Metrics per segment; ``predictions`` has y, p, cross_region and is_international."""
    out = {}
    for segment, flag in SEGMENTS.items():
        rows = predictions if flag is None else predictions[predictions[flag].astype(bool)]
        out[segment] = score(rows["y"].to_numpy(), rows["p"].to_numpy())
    return out


def reliability_plot(y: np.ndarray, p: np.ndarray, path: Path, title: str, bins: int = 10) -> Path:
    """Predicted vs observed win rate per probability bin; a perfect model lies on the diagonal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        mask = which == b
        if mask.any():
            xs.append(p[mask].mean())
            ys.append(y[mask].mean())
            ns.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect")
    ax.plot(xs, ys, marker="o", color="#2a6fdb", label="model")
    for x, yy, n in zip(xs, ys, ns, strict=True):
        ax.annotate(str(n), (x, yy), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set(xlabel="predicted P(win)", ylabel="observed win rate", xlim=(0, 1), ylim=(0, 1),
           title=title)  # fmt: skip
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
