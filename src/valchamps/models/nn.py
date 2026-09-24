"""PyTorch model that is symmetric by construction.

A small network g scores one team's view of a map. The map's logit is
``g(view from A) - g(view from B)``, so swapping the teams flips the sign exactly and
P(A wins) + P(B wins) = 1 without any averaging. Requires the optional ``nn`` extra (torch).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd

from valchamps.models.base import partner_index
from valchamps.models.dataset import NUMERIC_FEATURES, time_val_split
from valchamps.models.linear import numeric_preprocessor

DEFAULT_PARAMS: dict[str, Any] = {
    "hidden": 32,
    "map_embedding": 4,
    "dropout": 0.1,
    "lr": 0.003,
    "weight_decay": 0.001,
    "epochs": 300,
    "patience": 20,
    "batch_size": 256,
}


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the nn model needs PyTorch: install it with `uv sync --extra nn`"
        ) from exc
    return torch


class NNModel:
    name = "nn"

    def __init__(
        self, params: dict[str, Any] | None = None, val_fraction: float = 0.15, seed: int = 0
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.val_fraction = val_fraction
        self.seed = seed
        self.prep = numeric_preprocessor()
        self.maps_: dict[str, int] = {}
        self.n_inputs_: int | None = None
        self.net = None

    # --- pickling -----------------------------------------------------------------------------
    # The torch module class is defined inside _build (so torch is only imported when used),
    # which pickle can't serialise. Save the learned weights instead and rebuild on load.

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        net = state.pop("net")
        state["net_weights"] = (
            None
            if net is None
            else {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("net_weights", None)
        self.__dict__.update(state)
        self.net = None
        if weights is not None:
            torch = _torch()
            net = self._build(self.n_inputs_)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            net.eval()
            self.net = net

    # --- tensors ------------------------------------------------------------------------------

    def _map_ids(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.maps_.get(m, 0) for m in df["map_name"]])  # 0 = unseen map

    def _pairs(self, df: pd.DataFrame):
        """(x_row, x_partner, map_id) for every row that has its mirrored row in ``df``."""
        torch = _torch()
        partner = partner_index(df)
        keep = np.flatnonzero(partner >= 0)
        x = self.prep.transform(df[NUMERIC_FEATURES]).astype(np.float32)
        return (
            keep,
            torch.from_numpy(x[keep]),
            torch.from_numpy(x[partner[keep]]),
            torch.from_numpy(self._map_ids(df)[keep]),
        )

    # --- model --------------------------------------------------------------------------------

    def _build(self, n_inputs: int):
        torch = _torch()
        p = self.params

        class Scorer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = torch.nn.Embedding(n_maps, p["map_embedding"])
                self.g = torch.nn.Sequential(
                    torch.nn.Linear(n_inputs + p["map_embedding"], p["hidden"]),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p["dropout"]),
                    torch.nn.Linear(p["hidden"], p["hidden"]),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p["dropout"]),
                    torch.nn.Linear(p["hidden"], 1),
                )

            def forward(self, x_a, x_b, maps):
                e = self.embed(maps)
                score_a = self.g(torch.cat([x_a, e], dim=1)).squeeze(1)
                score_b = self.g(torch.cat([x_b, e], dim=1)).squeeze(1)
                return score_a - score_b  # logit of P(a wins)

        n_maps = len(self.maps_) + 1
        return Scorer()

    def fit(self, df: pd.DataFrame) -> NNModel:
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.maps_ = {m: i + 1 for i, m in enumerate(sorted(df["map_name"].unique()))}
        self.prep.fit(df[NUMERIC_FEATURES])

        train, val = time_val_split(df, self.val_fraction)
        # One example per map (perspective 0); antisymmetry covers the mirrored view.
        _, xa, xb, maps = self._pairs(train)
        y = torch.tensor(train["y"].to_numpy()[_first_views(train)], dtype=torch.float32)
        sel = torch.from_numpy(_first_view_mask(train))
        xa, xb, maps = xa[sel], xb[sel], maps[sel]
        has_val = len(val) > 0
        if has_val:
            _, vxa, vxb, vmaps = self._pairs(val)
            vsel = torch.from_numpy(_first_view_mask(val))
            vxa, vxb, vmaps = vxa[vsel], vxb[vsel], vmaps[vsel]
            vy = torch.tensor(val["y"].to_numpy()[_first_views(val)], dtype=torch.float32)

        self.n_inputs_ = int(xa.shape[1])
        net = self._build(self.n_inputs_)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, since_best = math.inf, None, 0
        batch = self.params["batch_size"]
        for _ in range(self.params["epochs"]):
            net.train()
            order = torch.from_numpy(rng.permutation(len(y)))
            for start in range(0, len(y), batch):
                idx = order[start : start + batch]
                opt.zero_grad()
                loss_fn(net(xa[idx], xb[idx], maps[idx]), y[idx]).backward()
                opt.step()
            if not has_val:
                continue
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(vxa, vxb, vmaps), vy).item()
            if val_loss < best - 1e-5:
                best, best_state, since_best = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                since_best += 1
                if since_best >= self.params["patience"]:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        torch = _torch()
        if self.net is None:
            raise RuntimeError("fit the model first")
        keep, xa, xb, maps = self._pairs(df)
        out = np.full(len(df), 0.5)
        with torch.no_grad():
            out[keep] = torch.sigmoid(self.net(xa, xb, maps)).numpy()
        return out


def _first_view_mask(df: pd.DataFrame) -> np.ndarray:
    """Among rows that have a partner, which are perspective 0."""
    partner = partner_index(df)
    return (df["perspective"].to_numpy() == 0)[partner >= 0]


def _first_views(df: pd.DataFrame) -> np.ndarray:
    partner = partner_index(df)
    keep = np.flatnonzero(partner >= 0)
    return keep[df["perspective"].to_numpy()[keep] == 0]
