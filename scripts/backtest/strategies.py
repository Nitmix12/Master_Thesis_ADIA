"""
Regime-driven portfolio weight rules (hard label + soft probability blends).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from scripts.backtest.loaders import (
    EW14_WEIGHT,
    EW_THREE_WEIGHT,
    INVESTABLE_14_COLS,
)
from scripts.backtest.signals import RegimeSignal, regime_index_sets

StrategyKind = Literal["single", "two", "three", "eq_sh_cash", "multi"]

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
    cash: pd.Series | None = None
    multi: pd.DataFrame | None = None
    asset_columns: tuple[str, ...] | None = None


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


# K=3 Bear / Neutral / Bull: partial equity in sidewalk (Neutral) state.
K3_EQUITY_SCORE_HARD: dict[int, float] = {0: 0.00, 1: 0.50, 2: 1.00}


def _k3_equity_score(signal: RegimeSignal, *, soft: bool) -> pd.Series:
    """Map K=3 regimes to [0, 1] equity tilt (Bear / half Neutral / full Bull)."""
    idx = signal.index
    sets = regime_index_sets(3)
    if soft:
        p_bull = _sum_probs(signal.probabilities, sets["bull"])
        p_neutral = _sum_probs(signal.probabilities, sets["steady"])
        return (p_bull + 0.5 * p_neutral).reindex(idx).fillna(0.0).clip(0.0, 1.0)
    return signal.regime_id.map(K3_EQUITY_SCORE_HARD).astype(float).reindex(idx).fillna(0.0)


def buy_and_hold(signal: RegimeSignal) -> StrategyWeights:
    """B0: 100% equity every month."""
    w = pd.Series(1.0, index=signal.index, name="equity")
    return StrategyWeights(kind="single", equity=w)


def buy_and_hold_ew_three(signal: RegimeSignal) -> StrategyWeights:
    """B1: equal-weight buy & hold on the three investable sleeves (1/3 each).

    SPXT (equity) + LUATTRUU (treasuries) + BCOMTR (commodities). Not EW17.
    """
    idx = signal.index
    w = EW_THREE_WEIGHT
    return StrategyWeights(
        kind="three",
        equity=pd.Series(w, index=idx),
        safe_haven=pd.Series(w, index=idx),
        commodity=pd.Series(w, index=idx),
    )


def buy_and_hold_ew14(signal: RegimeSignal) -> StrategyWeights:
    """B2: equal-weight buy & hold on the 14 TRS-investable factors (1/14 each)."""
    idx = signal.index
    w = EW14_WEIGHT
    multi = pd.DataFrame({col: w for col in INVESTABLE_14_COLS}, index=idx)
    return StrategyWeights(kind="multi", multi=multi, asset_columns=INVESTABLE_14_COLS)


def risk_on_off_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Risk-on regimes -> 100% equity; risk-off regimes -> 0% equity (cash).

    K=3: Bear 0%, Neutral 50%, Bull 100% (hard or probability blend).
    """
    idx = signal.index
    if signal.k == 3:
        w_eq = _k3_equity_score(signal, soft=soft)
        return StrategyWeights(kind="single", equity=w_eq)

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
    """Risk-on -> 100% SPXT; risk-off -> 100% LUATTRUU (via equity weight only).

    K=3: same equity score as risk-on/off; remainder in treasuries.
    """
    idx = signal.index
    if signal.k == 3:
        w_eq = _k3_equity_score(signal, soft=soft)
        return StrategyWeights(kind="two", equity=w_eq)

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

    K=3: Bull + Neutral -> equity; Bear -> treasuries. No inflation sleeve (empty
    ``inflation`` set); commodity weight stays 0 in hard mode.
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

    K=3: uses Bear / half-Neutral / Bull equity score (same as risk-on/off).
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)
    max_eq = float(np.clip(1.0 - bond_floor, 0.0, 1.0))

    if signal.k == 3:
        w_eq = _k3_equity_score(signal, soft=soft)
    elif soft:
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

    K=3: no inflation regime — commodity sleeve only receives the 20% floor
    (tilt from ``inflation`` probs is zero); weights renormalize to eq/sh/cm.
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


def _bear_probability(signal: RegimeSignal, *, soft: bool) -> pd.Series:
    """Probability mass on Bear / crisis (regime 0) for cash-shelter rules."""
    idx = signal.index
    sets = regime_index_sets(signal.k)
    if soft:
        return _sum_probs(signal.probabilities, sets["risk_off"]).reindex(idx).fillna(0.0).clip(0.0, 1.0)
    return signal.regime_id.isin(sets["risk_off"]).astype(float).reindex(idx).fillna(0.0)


def regime_cash_shelter_weights(
    signal: RegimeSignal,
    *,
    soft: bool = False,
    bond_floor: float = 0.30,
    bear_cash_min: float = 0.80,
) -> StrategyWeights:
    """
    Bond-floor tactical allocation with a fourth sleeve: cash (0% return).

    Non-equity allocation splits between LUATTRUU and cash:
    - Bear / high crisis probability: 80-100% of the defensive sleeve in cash
    - Otherwise: defensive sleeve in treasuries

    Equity uses the same bond-floor tactical rule as ``bond_floor_tactical``.
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)
    max_eq = float(np.clip(1.0 - bond_floor, 0.0, 1.0))

    if signal.k == 3:
        w_eq = _k3_equity_score(signal, soft=soft)
    elif soft:
        w_eq = _sum_probs(signal.probabilities, sets["risk_on"]).clip(0.0, 1.0)
    else:
        w_eq = signal.regime_id.isin(sets["risk_on"]).astype(float)

    w_eq = (max_eq * w_eq).reindex(idx).fillna(0.0).clip(0.0, 1.0)

    p_bear = _bear_probability(signal, soft=soft)
    # 0% cash in non-bear; 80-100% of defensive sleeve in cash as bear prob rises.
    cash_frac = p_bear * (bear_cash_min + (1.0 - bear_cash_min) * p_bear)
    defensive = 1.0 - w_eq
    w_cash = (defensive * cash_frac).reindex(idx).fillna(0.0).clip(0.0, 1.0)
    w_sh = (defensive - w_cash).reindex(idx).fillna(0.0).clip(0.0, 1.0)

    return StrategyWeights(
        kind="eq_sh_cash",
        equity=w_eq,
        safe_haven=w_sh,
        cash=w_cash,
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

    K=3 hard: same Bear / half-Neutral / Bull equity scores as risk-on/off.
    K=3 soft: convex transform of the K=3 equity score.
    """
    idx = signal.index
    sets = regime_index_sets(signal.k)

    if signal.k == 3:
        score = _k3_equity_score(signal, soft=soft)
        w_eq = score.pow(float(gamma)) if soft else score
        return StrategyWeights(kind="single", equity=w_eq.reindex(idx).fillna(0.0).clip(0.0, 1.0))

    if soft:
        p_on = _sum_probs(signal.probabilities, sets["risk_on"]).clip(0.0, 1.0)
        w_eq = p_on.pow(float(gamma))
    else:
        w_eq = signal.regime_id.isin(sets["risk_on"]).astype(float)

    return StrategyWeights(kind="single", equity=w_eq.reindex(idx).fillna(0.0).clip(0.0, 1.0))


DATA_DRIVEN_STRATEGY_K: dict[str, int] = {
    "data_driven_3": 3,
    "data_driven_4": 4,
    "data_driven_5": 5,
}


def data_driven_portfolio_weights(
    signal: RegimeSignal,
    portfolios: dict[int, np.ndarray],
    *,
    soft: bool = False,
) -> StrategyWeights:
    """
    Frozen MV-optimal weights (training 1971–1989 only) on (SPXT, LUATTRUU, BCOMTR).

    Hard: month t uses ``portfolios[regime_id]``.
    Soft: ``w_t = Σ_k p_k(t) · w_k`` from walk-forward regime probabilities.
    """
    idx = signal.index

    if soft:
        probs = signal.probabilities.reindex(idx).fillna(0.0)
        w_eq = pd.Series(0.0, index=idx)
        w_sh = pd.Series(0.0, index=idx)
        w_cm = pd.Series(0.0, index=idx)
        for regime_id, vec in portfolios.items():
            col = f"Prob_Regime{int(regime_id)}"
            if col not in probs.columns:
                continue
            p = probs[col].astype(float)
            w_eq = w_eq + p * float(vec[0])
            w_sh = w_sh + p * float(vec[1])
            w_cm = w_cm + p * float(vec[2])
        return StrategyWeights(
            kind="three",
            equity=w_eq.reindex(idx).fillna(EW_THREE_WEIGHT),
            safe_haven=w_sh.reindex(idx).fillna(EW_THREE_WEIGHT),
            commodity=w_cm.reindex(idx).fillna(EW_THREE_WEIGHT),
        )

    rid = signal.regime_id.reindex(idx).astype(int)

    def _weight(regime_id: int, asset_idx: int) -> float:
        vec = portfolios.get(int(regime_id))
        if vec is None:
            return EW_THREE_WEIGHT
        return float(vec[asset_idx])

    w_eq = rid.map(lambda r: _weight(r, 0)).astype(float)
    w_sh = rid.map(lambda r: _weight(r, 1)).astype(float)
    w_cm = rid.map(lambda r: _weight(r, 2)).astype(float)

    return StrategyWeights(
        kind="three",
        equity=w_eq.reindex(idx).fillna(EW_THREE_WEIGHT),
        safe_haven=w_sh.reindex(idx).fillna(EW_THREE_WEIGHT),
        commodity=w_cm.reindex(idx).fillna(EW_THREE_WEIGHT),
    )


def data_driven_multi_weights(
    signal: RegimeSignal,
    portfolios: dict[int, np.ndarray],
    asset_columns: tuple[str, ...],
    *,
    ew_fallback_weight: float,
    soft: bool = False,
) -> StrategyWeights:
    """
    Frozen MV-optimal weights on an N-asset investable universe.

    Hard: month t uses ``portfolios[regime_id]``.
    Soft: ``w_t = Σ_k p_k(t) · w_k`` from walk-forward regime probabilities.
    """
    idx = signal.index
    n_assets = len(asset_columns)

    if soft:
        probs = signal.probabilities.reindex(idx).fillna(0.0)
        weight_matrix = np.zeros((len(idx), n_assets), dtype=float)
        for regime_id, vec in portfolios.items():
            col = f"Prob_Regime{int(regime_id)}"
            if col not in probs.columns:
                continue
            p = probs[col].astype(float).values[:, None]
            weight_matrix = weight_matrix + p * np.asarray(vec, dtype=float)
        multi = pd.DataFrame(weight_matrix, index=idx, columns=list(asset_columns))
        row_sum = multi.sum(axis=1)
        zero_rows = row_sum <= 0
        if zero_rows.any():
            multi.loc[zero_rows] = ew_fallback_weight
    else:
        rid = signal.regime_id.reindex(idx).astype(int)

        def _weight(regime_id: int, asset_idx: int) -> float:
            vec = portfolios.get(int(regime_id))
            if vec is None:
                return ew_fallback_weight
            return float(vec[asset_idx])

        multi = pd.DataFrame(
            {
                col: rid.map(lambda r, i=i: _weight(r, i)).astype(float)
                for i, col in enumerate(asset_columns)
            },
            index=idx,
        )

    return StrategyWeights(kind="multi", multi=multi, asset_columns=asset_columns)


def _make_data_driven_builder(expected_k: int):
    def builder(signal: RegimeSignal, *, soft: bool = False) -> StrategyWeights:
        from scripts.portfolio_allocation import load_regime_portfolios

        if signal.k != expected_k:
            raise ValueError(
                f"data_driven_{expected_k} requires K={expected_k} signals, got K={signal.k}"
            )
        portfolios = load_regime_portfolios(expected_k)
        return data_driven_portfolio_weights(signal, portfolios, soft=soft)

    return builder


def _make_data_driven_multi_builder(
    expected_k: int,
    *,
    variant: str,
    asset_columns: tuple[str, ...],
    ew_fallback_weight: float,
    strategy_suffix: str,
):
    def builder(signal: RegimeSignal, *, soft: bool = False) -> StrategyWeights:
        from scripts.portfolio_allocation import load_regime_portfolios

        if signal.k != expected_k:
            raise ValueError(
                f"data_driven_{expected_k}_{strategy_suffix} requires K={expected_k} signals, "
                f"got K={signal.k}"
            )
        portfolios = load_regime_portfolios(expected_k, variant=variant)
        return data_driven_multi_weights(
            signal,
            portfolios,
            asset_columns,
            ew_fallback_weight=ew_fallback_weight,
            soft=soft,
        )

    return builder


def _make_data_driven_14_builder(expected_k: int):
    return _make_data_driven_multi_builder(
        expected_k,
        variant="14",
        asset_columns=INVESTABLE_14_COLS,
        ew_fallback_weight=EW14_WEIGHT,
        strategy_suffix="14",
    )


BENCHMARK_STRATEGY_KEYS = frozenset({
    "buy_and_hold",
    "buy_and_hold_ew_three",
    "buy_and_hold_ew14",
})

BENCHMARK_DISPLAY_NAMES: dict[str, str] = {
    "buy_and_hold": "B0 — SPXT buy & hold",
    "buy_and_hold_ew_three": "B1 — EW3 buy & hold (SPXT / LUATTRUU / BCOMTR)",
    "buy_and_hold_ew14": "B2 — EW14 buy & hold (14 TRS-investable factors; excl. VIX / USGG3M / LUACOAS)",
}

BENCHMARK_CURVE_LABELS: dict[str, str] = {
    "buy_and_hold": "B0 — SPXT B&H",
    "buy_and_hold_ew_three": "B1 — EW3 B&H",
}

DATA_DRIVEN_14_STRATEGY_K: dict[str, int] = {
    "data_driven_3_14": 3,
    "data_driven_4_14": 4,
    "data_driven_5_14": 5,
}

EW14_BENCHMARK_CURVE_LABELS: dict[str, str] = {
    "buy_and_hold_ew14": "B2 — EW14 B&H",
}

STRATEGY_BUILDERS = {
    "buy_and_hold": lambda sig, soft=False: buy_and_hold(sig),
    "buy_and_hold_ew_three": lambda sig, soft=False: buy_and_hold_ew_three(sig),
    "buy_and_hold_ew14": lambda sig, soft=False: buy_and_hold_ew14(sig),
    "risk_on_off": risk_on_off_weights,
    "safe_haven": safe_haven_weights,
    "all_weather": all_weather_weights,
    "inverse_vol": inverse_vol_weights,
    "defensive_safe_haven": defensive_safe_haven_weights,
    "crisis_cap_inverse_vol": crisis_cap_inverse_vol_weights,
    "bond_floor_tactical": bond_floor_tactical_weights,
    "all_weather_defensive": all_weather_defensive_weights,
    "convex_soft_risk_on": convex_soft_risk_on_weights,
    "regime_cash_shelter": regime_cash_shelter_weights,
    "data_driven_3": _make_data_driven_builder(3),
    "data_driven_4": _make_data_driven_builder(4),
    "data_driven_5": _make_data_driven_builder(5),
    "data_driven_3_14": _make_data_driven_14_builder(3),
    "data_driven_4_14": _make_data_driven_14_builder(4),
    "data_driven_5_14": _make_data_driven_14_builder(5),
}
