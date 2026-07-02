"""
Regime labeling for Botte & Bao-style GMM (pmr_paper).

Two Hungarian steps (same design as old/04_Bloomberg_v3):
1. **Economic labeling** — sign templates on cluster means/vols → regime names
2. **Temporal tracking** (walk-forward) — match components month-to-month in raw feature space

Templates are taken from the *old* pipeline that produced good plots, not v1's
derived/softened templates.

K=3: Bear, Neutral, Bull (literature-style 3-state equity taxonomy)
K=4: Crisis, Inflation, Steady State, Walking on Ice (Bloomberg v3)
K=5: adds Bull Market (old/05_GMM_5reg) — separates low-vol equity rallies from Steady State
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Regime metadata
# ---------------------------------------------------------------------------

REGIME_NAMES_3: Dict[int, str] = {
    0: "Bear",
    1: "Neutral",
    2: "Bull",
}

REGIME_NAMES_4: Dict[int, str] = {
    0: "Crisis",
    1: "Inflation",
    2: "Steady State",
    3: "Walking on Ice",
}

REGIME_NAMES_5: Dict[int, str] = {
    0: "Crisis",
    1: "Inflation",
    2: "Steady State",
    3: "Walking on Ice",
    4: "Bull Market",
}

REGIME_ORDER_3: Tuple[str, ...] = (
    "Bear",
    "Neutral",
    "Bull",
)

REGIME_ORDER_4: Tuple[str, ...] = (
    "Crisis",
    "Inflation",
    "Steady State",
    "Walking on Ice",
)

REGIME_ORDER_5: Tuple[str, ...] = (
    "Crisis",
    "Inflation",
    "Steady State",
    "Walking on Ice",
    "Bull Market",
)

# 3-regime templates — Bear / Neutral / Bull (equity-style literature taxonomy).
# Bear ≈ Crisis risk-off; Bull ≈ Steady State + rallies; Neutral ≈ sidewalk / mixed.
# Neutral uses mostly zero signs so it absorbs ambiguous middle states.
TEMPLATE_MEAN_3: Dict[str, Dict[str, int]] = {
    "SPXT": {"Bear": -1, "Neutral": 0, "Bull": 1},
    "VIX": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "LUACOAS": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "MXEF": {"Bear": -1, "Neutral": 0, "Bull": 1},
    "BCOMTR": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "LUATTRUU": {"Bear": 1, "Neutral": 0, "Bull": 0},
    "USGG3M": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "DXY": {"Bear": 1, "Neutral": 0, "Bull": 0},
    "LF98TRUU": {"Bear": -1, "Neutral": 0, "Bull": 1},
    "M1WOMOM": {"Bear": -1, "Neutral": 0, "Bull": 1},
    "M1WO000V": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "DBFXCARU": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "BCIT1T": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "NEIXCTAT": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "M1WOMVOL": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "M1WOSC": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "M1WOQU": {"Bear": 0, "Neutral": 0, "Bull": 0},
}

TEMPLATE_VOL_3: Dict[str, Dict[str, int]] = {
    "SPXT": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "VIX": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "LUACOAS": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "MXEF": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "BCOMTR": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "LUATTRUU": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "USGG3M": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "DXY": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "LF98TRUU": {"Bear": 1, "Neutral": 0, "Bull": -1},
    "M1WOMOM": {"Bear": 0, "Neutral": 0, "Bull": -1},
    "M1WO000V": {"Bear": 0, "Neutral": 0, "Bull": -1},
    "DBFXCARU": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "BCIT1T": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "NEIXCTAT": {"Bear": 0, "Neutral": 0, "Bull": 0},
    "M1WOMVOL": {"Bear": 1, "Neutral": 1, "Bull": -1},
    "M1WOSC": {"Bear": 0, "Neutral": 0, "Bull": -1},
    "M1WOQU": {"Bear": 0, "Neutral": 0, "Bull": 0},
}

BOOSTS_3: Dict[str, Dict[str, float]] = {
    "Bear": {"SPXT": 1.40, "VIX": 1.35, "LUACOAS": 1.30, "LF98TRUU": 1.15},
    "Neutral": {},
    "Bull": {
        "SPXT": 1.30,
        "VIX": 1.20,
        "LUACOAS": 1.15,
        "LF98TRUU": 1.15,
        "M1WOMOM": 1.25,
        "MXEF": 1.20,
    },
}

# 4-regime templates — old/04_Bloomberg_v3 (explicit; not sliced from 5-reg)
TEMPLATE_MEAN_4: Dict[str, Dict[str, int]] = {
    "SPXT": {"Crisis": -1, "Inflation": -1, "Steady State": 1, "Walking on Ice": 1},
    "VIX": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "LUACOAS": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "MXEF": {"Crisis": -1, "Inflation": 0, "Steady State": 0, "Walking on Ice": 1},
    "BCOMTR": {"Crisis": -1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 0},
    "LUATTRUU": {"Crisis": 1, "Inflation": -1, "Steady State": 1, "Walking on Ice": 0},
    "USGG3M": {"Crisis": -1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 0},
    "DXY": {"Crisis": 1, "Inflation": -1, "Steady State": 0, "Walking on Ice": 0},
    "LF98TRUU": {"Crisis": -1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 0},
    "M1WOMOM": {"Crisis": 0, "Inflation": 0, "Steady State": 1, "Walking on Ice": -1},
    "M1WO000V": {"Crisis": 1, "Inflation": 1, "Steady State": 1, "Walking on Ice": 1},
    "DBFXCARU": {"Crisis": -1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 0},
    "BCIT1T": {"Crisis": -1, "Inflation": 1, "Steady State": 1, "Walking on Ice": 0},
    "NEIXCTAT": {"Crisis": 1, "Inflation": 0, "Steady State": 0, "Walking on Ice": 0},
    "M1WOMVOL": {"Crisis": 1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 1},
    "M1WOSC": {"Crisis": -1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 1},
    "M1WOQU": {"Crisis": 1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 1},
}

TEMPLATE_VOL_4: Dict[str, Dict[str, int]] = {
    "SPXT": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "VIX": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "LUACOAS": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "MXEF": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "BCOMTR": {"Crisis": 1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1},
    "LUATTRUU": {"Crisis": 1, "Inflation": 1, "Steady State": -1, "Walking on Ice": 1},
    "USGG3M": {"Crisis": 1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1},
    "DXY": {"Crisis": 1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1},
    "LF98TRUU": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "M1WOMOM": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "M1WO000V": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "DBFXCARU": {"Crisis": 1, "Inflation": 0, "Steady State": 0, "Walking on Ice": 1},
    "BCIT1T": {"Crisis": 1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1},
    "NEIXCTAT": {"Crisis": 1, "Inflation": 0, "Steady State": 0, "Walking on Ice": 1},
    "M1WOMVOL": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "M1WOSC": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
    "M1WOQU": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1},
}

BOOSTS_4: Dict[str, Dict[str, float]] = {}

# 5-regime templates — old/05_GMM_5reg
TEMPLATE_MEAN_5: Dict[str, Dict[str, int]] = {
    "SPXT": {"Crisis": -1, "Inflation": -1, "Steady State": 1, "Walking on Ice": 1, "Bull Market": 1},
    "VIX": {"Crisis": 1, "Inflation": 1, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "LUACOAS": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 0, "Bull Market": -1},
    "MXEF": {"Crisis": -1, "Inflation": 0, "Steady State": 0, "Walking on Ice": 1, "Bull Market": 1},
    "BCOMTR": {"Crisis": -1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 0, "Bull Market": -1},
    "LUATTRUU": {"Crisis": 1, "Inflation": -1, "Steady State": 1, "Walking on Ice": 0, "Bull Market": 0},
    "USGG3M": {"Crisis": -1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 0, "Bull Market": -1},
    "DXY": {"Crisis": 1, "Inflation": -1, "Steady State": 0, "Walking on Ice": 0, "Bull Market": 0},
    "LF98TRUU": {"Crisis": -1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 0, "Bull Market": 1},
    "M1WOMOM": {"Crisis": 0, "Inflation": 0, "Steady State": 0, "Walking on Ice": -1, "Bull Market": 1},
    "M1WO000V": {"Crisis": 1, "Inflation": 1, "Steady State": 1, "Walking on Ice": 1, "Bull Market": 0},
    "DBFXCARU": {"Crisis": -1, "Inflation": 1, "Steady State": 0, "Walking on Ice": 0, "Bull Market": 0},
    "BCIT1T": {"Crisis": -1, "Inflation": 1, "Steady State": 1, "Walking on Ice": 0, "Bull Market": 1},
    "NEIXCTAT": {"Crisis": 1, "Inflation": 0, "Steady State": 0, "Walking on Ice": 0, "Bull Market": 0},
    "M1WOMVOL": {"Crisis": 1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 1, "Bull Market": 0},
    "M1WOSC": {"Crisis": -1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 1, "Bull Market": 0},
    "M1WOQU": {"Crisis": 1, "Inflation": 0, "Steady State": 1, "Walking on Ice": 1, "Bull Market": 0},
}

TEMPLATE_VOL_5: Dict[str, Dict[str, int]] = {
    "SPXT": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "VIX": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "LUACOAS": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "MXEF": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "BCOMTR": {"Crisis": 0, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1, "Bull Market": -1},
    "LUATTRUU": {"Crisis": 0, "Inflation": 1, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "USGG3M": {"Crisis": 0, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1, "Bull Market": -1},
    "DXY": {"Crisis": 0, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1, "Bull Market": -1},
    "LF98TRUU": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "M1WOMOM": {"Crisis": 0, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "M1WO000V": {"Crisis": 0, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "DBFXCARU": {"Crisis": 0, "Inflation": 0, "Steady State": 0, "Walking on Ice": 1, "Bull Market": -1},
    "BCIT1T": {"Crisis": 0, "Inflation": 1, "Steady State": 0, "Walking on Ice": 1, "Bull Market": -1},
    "NEIXCTAT": {"Crisis": 0, "Inflation": 0, "Steady State": 0, "Walking on Ice": 1, "Bull Market": -1},
    "M1WOMVOL": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "M1WOSC": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
    "M1WOQU": {"Crisis": 1, "Inflation": 0, "Steady State": -1, "Walking on Ice": 1, "Bull Market": -1},
}

BOOSTS_5: Dict[str, Dict[str, float]] = {
    "Crisis": {"SPXT": 1.35, "VIX": 1.20, "LF98TRUU": 1.20},
    "Inflation": {"BCOMTR": 1.30, "USGG3M": 1.25, "LUATTRUU": 1.25},
}

ANCHOR_WINDOWS_4: List[Tuple] = [
    ("Oil Crisis / Stagflation", "1973-01", "1974-12", "Inflation", 0.40),
    ("Volcker era", "1979-01", "1982-12", "Inflation", 0.40),
    ("Black Monday", "1987-09", "1987-12", "Crisis", 0.25),
    ("GFC", "2007-07", "2009-03", "Crisis", 0.30),
    ("COVID crash", "2020-02", "2020-04", "Crisis", 0.33),
    ("Dot-com / late 90s", "1995-01", "1999-12", "Steady State", 0.30),
    ("Post-GFC expansion", "2012-01", "2019-12", "Steady State", 0.40),
]

ANCHOR_WINDOWS_5: List[Tuple] = [
    ("Oil Crisis / Stagflation", "1973-01", "1974-12", "Inflation", 0.40),
    ("Volcker era", "1979-01", "1982-12", "Inflation", 0.40),
    ("Black Monday", "1987-09", "1987-12", "Crisis", 0.25),
    ("GFC", "2007-07", "2009-03", "Crisis", 0.30),
    ("COVID crash", "2020-02", "2020-04", "Crisis", 0.33),
    ("Dot-com / late 90s", "1995-01", "1999-12", "Bull Market", 0.30),
    ("Post-GFC expansion", "2012-01", "2019-12", "Steady State", 0.35),
]

ANCHOR_WINDOWS_3: List[Tuple] = [
    ("Oil Crisis / Stagflation", "1973-01", "1974-12", "Bear", 0.35),
    ("Volcker era", "1979-01", "1982-12", "Bear", 0.35),
    ("Black Monday", "1987-09", "1987-12", "Bear", 0.25),
    ("GFC", "2007-07", "2009-03", "Bear", 0.30),
    ("COVID crash", "2020-02", "2020-04", "Bear", 0.33),
    ("Dot-com / late 90s", "1995-01", "1999-12", "Bull", 0.30),
    ("Post-GFC expansion", "2012-01", "2019-12", "Bull", 0.35),
]

REGIME_COLORS: Dict[int, str] = {
    0: "purple",
    1: "red",
    2: "lightblue",
    3: "yellow",
    4: "darkgreen",
}

REGIME_COLORS_3: Dict[int, str] = {
    0: "purple",
    1: "yellow",
    2: "darkgreen",
}

REGIME_COLORS_4: Dict[int, str] = {
    0: "purple",
    1: "red",
    2: "lightblue",
    3: "yellow",
}

REGIME_COLORS_5: Dict[int, str] = {
    0: "purple",
    1: "red",
    2: "lightblue",
    3: "yellow",
    4: "darkgreen",
}


@dataclass(frozen=True)
class RegimeSpec:
    k: int
    names: Dict[int, str]
    order: Tuple[str, ...]
    template_mean: Dict[str, Dict[str, int]]
    template_vol: Dict[str, Dict[str, int]]
    boosts: Dict[str, Dict[str, float]]
    anchor_windows: List[Tuple]
    colors: Dict[int, str]


def get_spec(k: int) -> RegimeSpec:
    if k == 3:
        return RegimeSpec(
            k=3,
            names=REGIME_NAMES_3,
            order=REGIME_ORDER_3,
            template_mean=TEMPLATE_MEAN_3,
            template_vol=TEMPLATE_VOL_3,
            boosts=BOOSTS_3,
            anchor_windows=ANCHOR_WINDOWS_3,
            colors=REGIME_COLORS_3,
        )
    if k == 4:
        return RegimeSpec(
            k=4,
            names=REGIME_NAMES_4,
            order=REGIME_ORDER_4,
            template_mean=TEMPLATE_MEAN_4,
            template_vol=TEMPLATE_VOL_4,
            boosts=BOOSTS_4,
            anchor_windows=ANCHOR_WINDOWS_4,
            colors=REGIME_COLORS_4,
        )
    if k == 5:
        return RegimeSpec(
            k=5,
            names=REGIME_NAMES_5,
            order=REGIME_ORDER_5,
            template_mean=TEMPLATE_MEAN_5,
            template_vol=TEMPLATE_VOL_5,
            boosts=BOOSTS_5,
            anchor_windows=ANCHOR_WINDOWS_5,
            colors=REGIME_COLORS_5,
        )
    raise ValueError(f"Only K=3, K=4, or K=5 supported; got K={k}")


def regime_names(k: int) -> Dict[int, str]:
    return get_spec(k).names


def _feature_idx(cols: pd.Index, name: str) -> Optional[int]:
    try:
        return list(cols).index(name)
    except ValueError:
        return None


def _diag_variances(model: Any) -> np.ndarray:
    cov = np.asarray(model.covariances_, dtype=float)
    n_components = int(getattr(model, "n_components"))
    if cov.ndim == 3:
        return np.diagonal(cov, axis1=1, axis2=2)
    if cov.ndim == 2:
        if cov.shape[0] == n_components:
            return cov
        if cov.shape[0] == cov.shape[1]:
            return np.tile(np.diag(cov), (n_components, 1))
        return cov
    raise ValueError(f"Unexpected covariances_ shape: {cov.shape}")


def _inverse_scale_weights(model: Any) -> np.ndarray:
    vars_kf = np.clip(_diag_variances(model), 1e-12, None)
    scale = np.sqrt(np.mean(vars_kf, axis=0))
    return 1.0 / np.maximum(scale, 1e-12)


def compute_scores(model: Any, feature_columns: pd.Index, k: int) -> np.ndarray:
    """Return (K, k) template match scores (higher = better)."""
    spec = get_spec(k)
    means = np.asarray(model.means_, dtype=float)
    vol = np.sqrt(np.clip(_diag_variances(model), 1e-12, None))
    weights = _inverse_scale_weights(model)
    scores = np.zeros((means.shape[0], spec.k), dtype=float)

    for factor in spec.template_mean:
        fi = _feature_idx(feature_columns, factor)
        if fi is None:
            continue
        w_f = float(weights[fi])
        for r_idx, regime_name in enumerate(spec.order):
            s_m = int(spec.template_mean[factor][regime_name])
            s_v = int(spec.template_vol[factor][regime_name])
            boost = float(spec.boosts.get(regime_name, {}).get(factor, 1.0))
            scores[:, r_idx] += (w_f * boost) * (s_m * means[:, fi] + s_v * vol[:, fi])
    return scores


def map_clusters(model: Any, feature_columns: pd.Index, k: Optional[int] = None) -> Dict[int, int]:
    """
    Hungarian assignment: raw sklearn component id → canonical regime id (0..k-1).
    """
    n_components = int(getattr(model, "n_components"))
    if k is None:
        k = n_components
    if k != n_components:
        raise ValueError(f"map_clusters: model has K={n_components} but k={k}")

    scores = compute_scores(model, feature_columns, k)
    row_ind, col_ind = linear_sum_assignment(-scores)
    return {int(raw): int(regime) for raw, regime in zip(row_ind, col_ind)}


def relabel_states(raw_states: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    return np.array([mapping[int(s)] for s in np.asarray(raw_states)], dtype=int)


def relabel_state_probabilities(raw_probs: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    probs = np.asarray(raw_probs, dtype=float)
    k = probs.shape[1]
    out = np.zeros_like(probs)
    for raw_state, regime_id in mapping.items():
        if raw_state < k and regime_id < k:
            out[:, regime_id] = probs[:, raw_state]
    return out


def hungarian_align_means(
    new_means_raw: np.ndarray,
    prev_means_raw: np.ndarray,
    feature_weights: np.ndarray,
) -> np.ndarray:
    """
    Temporal Hungarian: canon_to_sklearn[j] = sklearn index for canonical slot j.
    Means must be in the same coordinate system (raw feature space after inverse_transform).
    """
    k = new_means_raw.shape[0]
    w = feature_weights.reshape(1, -1)
    cost = np.zeros((k, k), dtype=float)
    for i in range(k):
        diff = new_means_raw[i] - prev_means_raw
        cost[i, :] = np.sqrt(np.sum(w * diff * diff, axis=1))
    _, col_ind = linear_sum_assignment(cost)
    canon_to_sklearn = np.empty(k, dtype=int)
    for m in range(k):
        canon_to_sklearn[col_ind[m]] = m
    return canon_to_sklearn


def tracker_feature_weights(columns: pd.Index) -> np.ndarray:
    w = np.ones(len(columns), dtype=float)
    boost = {
        "SPXT": 3.0,
        "VIX": 3.0,
        "LUACOAS": 3.0,
        "LF98TRUU": 3.0,
        "BCOMTR": 2.0,
        "USGG3M": 2.0,
        "LUATTRUU": 2.0,
        "DXY": 1.5,
        "MXEF": 1.5,
        "M1WOMOM": 1.5,
        "M1WOMVOL": 1.5,
    }
    for name, val in boost.items():
        if name in columns:
            w[list(columns).index(name)] = val
    return w


def print_score_diagnostics(
    model: Any, feature_columns: pd.Index, mapping: Dict[int, int], k: int
) -> None:
    spec = get_spec(k)
    scores = compute_scores(model, feature_columns, k)
    cols = "  ".join(f"{r:>12s}" for r in spec.order)
    print(f"\n--- Template scores (row=raw cluster, col=regime) ---\n{'Cluster':>8}  {cols}  {'Assigned':>14}")
    for i in range(scores.shape[0]):
        row = "  ".join(f"{scores[i, j]:12.3f}" for j in range(scores.shape[1]))
        print(f"{i:8d}  {row}  {spec.names[mapping[i]]:>14s}")
    print()


def sanity_check(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    k: int,
    *,
    silent: bool = False,
) -> bool:
    spec = get_spec(k)
    all_pass = True
    lines = ["", "=" * 72, f"  SANITY CHECK (K={k})", "=" * 72]
    for name, start, end, expected, min_share in spec.anchor_windows:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        n = int(mask.sum())
        if n == 0:
            lines.append(f"  {name:30s}  SKIPPED")
            continue
        reg_id = next(r for r, nm in spec.names.items() if nm == expected)
        share = float((labels[mask] == reg_id).sum() / n)
        ok = share >= min_share
        all_pass = all_pass and ok
        counts = Counter(labels[mask])
        detail = ", ".join(f"{spec.names[r]}:{c}" for r, c in sorted(counts.items()))
        lines.append(
            f"  [{'PASS' if ok else 'FAIL'}] {name:30s}  "
            f"want {expected:>14s} >= {min_share:.0%}  |  got {share:.0%}  ({detail})"
        )
    lines += [
        "=" * 72,
        f"  Result: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}",
        "=" * 72,
    ]
    if not silent:
        print("\n".join(lines))
    return all_pass
