"""
Regime-driven portfolio weight rules (hard label + soft probability blends).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from scripts.backtest.signals import RegimeSignal, regime_index_sets

StrategyKind = Literal["single", "two", "three"]

# Equity exposure scaling for inverse-vol heuristic (regime-based vol targeting).
INVERSE_VOL_SCALES: dict[int, float] = {
    0: 0.25,  # Crisis / Bear
    1: 0.75,  # Inflation / Neutral
    2: 1.00,  # Steady State / Bull
    3: 0.50,  # Walking on Ice
    4: 1.00,  # Bull Market
}


@dataclass
class StrategyWeights:
    kind: StrategyKind
    equity: pd.Series | None = None
    safe_haven: pd.Series | None = None
    commodity: pd.Series | None = None


def _sum_probs(probs: pd.DataFrame, regime_ids: list[int]) -> pd.Series:
    cols = [f"Prob_Regime{i}" for i in regime_ids]
    existing = [c for c in cols if c in probs.columns]
    if not existing:
        return pd.Series(0.0, index=probs.index)
    return probs[existing].sum(axis=1)


def _normalize_three(
    w_eq: pd.Series, w_sh: pd.Series, w_cm: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    total = w_eq + w_sh + w_cm
    total = total.replace(0.0, np.nan)
    w_eq = (w_eq / total).fillna(0.0)
    w_sh = (w_sh / total).fillna(0.0)
    w_cm = (w_cm / total).fillna(0.0)
    return w_eq, w_sh, w_cm


def buy_and_hold(signal: RegimeSignal) -> StrategyWeights:
    """B0: 100% equity every month."""
    w = pd.Series(1.0, index=signal.index, name="equity")
    return StrategyWeights(kind="single", equity=w)


def buy_and_hold_ew_three(signal: RegimeSignal) -> StrategyWeights:
    """B1: 1/3 SPXT, 1/3 treasuries, 1/3 commodities every month."""
    idx = signal.index
    third = 1.0 / 3.0
    return StrategyWeights(
        kind="three",
        equity=pd.Series(third, index=idx),
        safe_haven=pd.Series(third, index=idx),
        commodity=pd.Series(third, index=idx),
    )


def risk_on_off_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Risk-on regimes -> 100% equity; risk-off regimes -> 0% equity (cash).
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)

    if soft:
        w_eq = _sum_probs(signal.probabilities, sets["risk_on"]).clip(0.0, 1.0)
    else:
        w_eq = signal.regime_id.isin(sets["risk_on"]).astype(float)

    w_eq = w_eq.reindex(idx).fillna(0.0)
    return StrategyWeights(kind="single", equity=w_eq)


def safe_haven_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """Risk-on -> 100% SPXT; risk-off -> 100% LUATTRUU (via equity weight only)."""
    idx = signal.index
    sets = regime_index_sets(signal.k)

    if soft:
        w_eq = _sum_probs(signal.probabilities, sets["risk_on"]).clip(0.0, 1.0)
    else:
        w_eq = signal.regime_id.isin(sets["risk_on"]).astype(float)

    w_eq = w_eq.reindex(idx).fillna(0.0)
    return StrategyWeights(kind="two", equity=w_eq)


def all_weather_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Growth-like regimes -> equity; defensive regimes -> treasuries; inflation -> commodities.
    K=3: Bull + Neutral -> equity; Bear -> treasuries (no inflation sleeve).
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)
    r = signal.regime_id

    if soft:
        p = signal.probabilities
        w_eq = _sum_probs(p, sets["steady"] + sets["bull"])
        w_sh = _sum_probs(p, sets["risk_off"])
        w_cm = _sum_probs(p, sets["inflation"])
        w_eq, w_sh, w_cm = _normalize_three(w_eq, w_sh, w_cm)
    else:
        w_eq = r.isin(sets["steady"] + sets["bull"]).astype(float)
        w_sh = r.isin(sets["risk_off"]).astype(float)
        w_cm = r.isin(sets["inflation"]).astype(float)
        # Exactly one regime active per month; no normalization needed.

    return StrategyWeights(
        kind="three",
        equity=w_eq.reindex(idx).fillna(0.0),
        safe_haven=w_sh.reindex(idx).fillna(0.0),
        commodity=w_cm.reindex(idx).fillna(0.0),
    )


def inverse_vol_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
    scales: dict[int, float] | None = None,
) -> StrategyWeights:
    """Regime-scaled equity exposure (inverse-vol style heuristic)."""
    if scales is None:
        scales = (
            {0: 0.25, 1: 0.50, 2: 1.00}
            if signal.k == 3
            else INVERSE_VOL_SCALES
        )
    idx = signal.index

    if soft:
        exposure = pd.Series(0.0, index=idx)
        for reg_id, scale in scales.items():
            col = f"Prob_Regime{reg_id}"
            if col in signal.probabilities.columns:
                exposure = exposure + signal.probabilities[col] * float(scale)
    else:
        exposure = signal.regime_id.map(scales).astype(float)

    exposure = exposure.reindex(idx).fillna(1.0).clip(0.0, 1.0)
    return StrategyWeights(kind="single", equity=exposure)


def defensive_safe_haven_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Two-asset defensive rotation:
    - Crisis/WOI/Bear: 0% equity
    - Inflation/Neutral: partial equity
    - Steady/Bull: higher equity
    """
    idx = signal.index
    r = signal.regime_id
    # Lower equity in inflation than pure risk-on allocation.
    scales: dict[int, float] = (
        {0: 0.00, 1: 0.50, 2: 1.00}
        if signal.k == 3
        else {0: 0.00, 1: 0.35, 2: 0.75, 3: 0.00, 4: 1.00}
    )

    if soft:
        w_eq = pd.Series(0.0, index=idx)
        for reg_id, scale in scales.items():
            col = f"Prob_Regime{reg_id}"
            if col in signal.probabilities.columns:
                w_eq = w_eq + signal.probabilities[col] * float(scale)
    else:
        w_eq = r.map(scales).astype(float)

    return StrategyWeights(kind="two", equity=w_eq.reindex(idx).fillna(0.0).clip(0.0, 1.0))


def crisis_cap_inverse_vol_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Inverse-vol exposure with an explicit crisis-probability cap.
    Useful to de-risk soft allocations during left-tail events.
    """
    idx = signal.index
    base = inverse_vol_weights(signal, soft=soft).equity
    assert base is not None

    p_crisis = (
        signal.probabilities.get("Prob_Regime0", pd.Series(0.0, index=idx))
        .reindex(idx)
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    # When crisis prob rises, cap equity rapidly.
    crisis_cap = (1.0 - 1.25 * p_crisis).clip(lower=0.10, upper=1.0)
    w_eq = np.minimum(base.values, crisis_cap.values)
    w_eq = pd.Series(w_eq, index=idx).clip(0.0, 1.0)
    return StrategyWeights(kind="single", equity=w_eq)


def bond_floor_tactical_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
    bond_floor: float = 0.30,
) -> StrategyWeights:
    """
    Two-asset tactical allocation with a permanent treasury floor.
    Equity is the risk-on signal but never exceeds 1 - bond_floor.
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)
    max_eq = float(np.clip(1.0 - bond_floor, 0.0, 1.0))

    if soft:
        w_eq = _sum_probs(signal.probabilities, sets["risk_on"]).clip(0.0, 1.0)
    else:
        w_eq = signal.regime_id.isin(sets["risk_on"]).astype(float)

    w_eq = (max_eq * w_eq).reindex(idx).fillna(0.0).clip(0.0, 1.0)
    return StrategyWeights(kind="two", equity=w_eq)


def all_weather_defensive_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Three-asset defensive all-weather:
    start from equal-risk-style floors, tilt by regime probabilities.
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)

    if soft:
        p = signal.probabilities
        p_eq = _sum_probs(p, sets["steady"] + sets["bull"])
        p_sh = _sum_probs(p, sets["risk_off"])
        p_cm = _sum_probs(p, sets["inflation"])
    else:
        r = signal.regime_id
        p_eq = r.isin(sets["steady"] + sets["bull"]).astype(float)
        p_sh = r.isin(sets["risk_off"]).astype(float)
        p_cm = r.isin(sets["inflation"]).astype(float)

    # Floor + tilt: keeps diversification even when one regime dominates.
    w_eq = 0.20 + 0.60 * p_eq
    w_sh = 0.20 + 0.60 * p_sh
    w_cm = 0.20 + 0.60 * p_cm
    w_eq, w_sh, w_cm = _normalize_three(w_eq, w_sh, w_cm)

    return StrategyWeights(
        kind="three",
        equity=w_eq.reindex(idx).fillna(0.0),
        safe_haven=w_sh.reindex(idx).fillna(0.0),
        commodity=w_cm.reindex(idx).fillna(0.0),
    )


def convex_soft_risk_on_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
    gamma: float = 1.50,
) -> StrategyWeights:
    """
    Convex risk-on mapping for soft mode:
    equity = p(risk_on)^gamma, gamma>1 -> more defensive at middling confidence.
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)

    if soft:
        p_on = _sum_probs(signal.probabilities, sets["risk_on"]).clip(0.0, 1.0)
        w_eq = p_on.pow(float(gamma))
    else:
        w_eq = signal.regime_id.isin(sets["risk_on"]).astype(float)

    return StrategyWeights(kind="single", equity=w_eq.reindex(idx).fillna(0.0).clip(0.0, 1.0))


BENCHMARK_STRATEGY_KEYS = frozenset({"buy_and_hold", "buy_and_hold_ew_three"})

STRATEGY_BUILDERS = {
    "buy_and_hold": lambda sig, soft=False: buy_and_hold(sig),
    "buy_and_hold_ew_three": lambda sig, soft=False: buy_and_hold_ew_three(sig),
    "risk_on_off": risk_on_off_weights,
    "safe_haven": safe_haven_weights,
    "all_weather": all_weather_weights,
    "inverse_vol": inverse_vol_weights,
    "defensive_safe_haven": defensive_safe_haven_weights,
    "crisis_cap_inverse_vol": crisis_cap_inverse_vol_weights,
    "bond_floor_tactical": bond_floor_tactical_weights,
    "all_weather_defensive": all_weather_defensive_weights,
    "convex_soft_risk_on": convex_soft_risk_on_weights,
}
