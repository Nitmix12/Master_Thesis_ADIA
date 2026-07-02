"""Align walk-forward regime signals to return dates (causal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from scripts.backtest.loaders import load_regime_backtest_bundle


@dataclass
class RegimeSignal:
    """Regime hard labels and posterior probabilities on backtest dates."""

    k: int
    index: pd.DatetimeIndex
    regime_id: pd.Series
    regime_name: pd.Series
    probabilities: pd.DataFrame

    @property
    def n_regimes(self) -> int:
        return self.k


def probability_columns(k: int) -> list[str]:
    return [f"Prob_Regime{i}" for i in range(k)]


def load_walk_forward_signals(
    k: int,
    *,
    test_start: str = "1990-01-31",
) -> RegimeSignal:
    """Load returns-aligned walk-forward probabilities and hard labels."""
    panel, wf, regime = load_regime_backtest_bundle(k, test_start=test_start)
    prob_cols = probability_columns(k)
    missing = [c for c in prob_cols if c not in wf.columns]
    if missing:
        raise ValueError(f"walk_forward_k{k} missing columns: {missing}")

    probs = wf[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probs = probs.div(probs.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    names = wf["Regime_Name"] if "Regime_Name" in wf.columns else regime.astype(str)

    return RegimeSignal(
        k=k,
        index=panel.index,
        regime_id=regime,
        regime_name=names,
        probabilities=probs,
    )


def regime_index_sets(k: int) -> Dict[str, list[int]]:
    """Canonical regime ids grouped for strategy rules."""
    if k == 3:
        return {
            "crisis": [0],  # Defensive
            "inflation": [1],
            "steady": [],
            "woi": [],
            "bull": [2],  # Growth
            "risk_on": [2],
            "risk_off": [0],
        }

    base = {
        "crisis": [0],
        "inflation": [1],
        "steady": [2],
        "woi": [3],
    }
    if k == 5:
        base["bull"] = [4]
        base["risk_on"] = [1, 2, 4]
    else:
        base["bull"] = []
        base["risk_on"] = [1, 2]
    base["risk_off"] = [0, 3]
    return base
