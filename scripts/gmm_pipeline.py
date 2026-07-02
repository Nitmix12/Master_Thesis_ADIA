"""
Static and walk-forward GMM with old-pipeline hyperparameters + regime labeling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from scripts.regime_labeling import (
    hungarian_align_means,
    map_clusters,
    relabel_states,
    sanity_check,
    tracker_feature_weights,
)


def _gmm_config(k: int) -> dict:
    """GMM fit settings (old/04_Bloomberg_v3 + old/05_GMM_5reg)."""
    if k == 3:
        return dict(
            covariance_type="full",
            n_init=20,
            max_iter=500,
            reg_covar=1e-5,
        )
    if k == 4:
        return dict(
            covariance_type="full",
            n_init=20,
            max_iter=500,
            reg_covar=1e-5,
        )
    if k == 5:
        return dict(
            covariance_type="diag",
            n_init=20,
            max_iter=500,
            reg_covar=1e-6,
        )
    raise ValueError(f"K must be 3, 4, or 5; got {k}")


def _wf_smoothing_config(k: int) -> Tuple[int, float]:
    """
    Layer 1: moderate EMA — responsive but trims 1-month probability spikes.

    No long spans (12–18). K=5 uses a longer span and lower raw weight than K=4
    to limit five-way label flicker (island suppression is off on the trading path).
    """
    if k in {3, 4}:
        return 3, 0.60
    if k == 5:
        return 9, 0.33
    raise ValueError(f"K must be 3, 4, or 5; got {k}")


CRISIS_REGIME_ID = 0


def _calm_regime_ids(k: int) -> frozenset[int]:
    """Non-crisis regimes that require confirmation before entry."""
    if k == 3:
        return frozenset({1, 2})
    if k == 4:
        return frozenset({1, 2, 3})
    if k == 5:
        return frozenset({1, 2, 3, 4})
    raise ValueError(f"K must be 3, 4, or 5; got {k}")


def _candidate_streak(candidates: np.ndarray, t: int, regime_id: int) -> int:
    streak = 0
    for s in range(t, -1, -1):
        if candidates[s] == regime_id:
            streak += 1
        else:
            break
    return streak


def _required_dwell_months(
    target: int,
    current: int,
    *,
    k: int,
    calm_dwell_months: int,
    crisis_enter_months: int,
    crisis_exit_months: int,
    crisis_prob: float,
    crisis_strong_prob: float,
    crisis_enter_weak_months: int,
) -> int:
    if target == CRISIS_REGIME_ID:
        if crisis_prob >= crisis_strong_prob:
            return crisis_enter_months
        return crisis_enter_weak_months
    if current == CRISIS_REGIME_ID:
        return crisis_exit_months
    if target in _calm_regime_ids(k):
        return calm_dwell_months
    return 1


def apply_dwell_hysteresis(
    candidates: np.ndarray,
    crisis_probs: np.ndarray,
    *,
    k: int,
    calm_dwell_months: int = 2,
    crisis_enter_months: int = 1,
    crisis_exit_months: int = 1,
    crisis_enter_weak_months: int = 2,
    crisis_strong_prob: float = 0.32,
) -> np.ndarray:
    """
    Layer 2: asymmetric persistence on argmax candidates.

    - Enter calm: ``calm_dwell_months`` consecutive dominant months
    - Exit crisis: ``crisis_exit_months`` (fast — avoids multi-year purple blocks)
    - Enter crisis: 1 month if blended crisis prob is high, else 2 months
    """
    candidates = np.asarray(candidates, dtype=int)
    crisis_probs = np.asarray(crisis_probs, dtype=float)
    n = len(candidates)
    if n == 0:
        return candidates.copy()
    if len(crisis_probs) != n:
        raise ValueError("candidates and crisis_probs must have equal length")

    out = np.empty(n, dtype=int)
    current = int(candidates[0])
    out[0] = current

    for t in range(1, n):
        cand = int(candidates[t])
        if cand == current:
            out[t] = current
            continue

        required = _required_dwell_months(
            cand,
            current,
            k=k,
            calm_dwell_months=calm_dwell_months,
            crisis_enter_months=crisis_enter_months,
            crisis_exit_months=crisis_exit_months,
            crisis_prob=float(crisis_probs[t]),
            crisis_strong_prob=crisis_strong_prob,
            crisis_enter_weak_months=crisis_enter_weak_months,
        )
        if _candidate_streak(candidates, t, cand) >= required:
            current = cand
        out[t] = current

    return out


def suppress_island_flickers(
    labels: np.ndarray,
    crisis_probs: np.ndarray,
    *,
    max_island_len: int = 2,
    crisis_keep_prob: float = 0.28,
) -> np.ndarray:
    """
    Layer 3 (optional): drop 1–2 month regime islands sandwiched by the same
    neighbour. Not applied in the default walk-forward path (``use_island_suppression``
    defaults to False) because confirming a sandwich requires a future month and
    retroactively revises labels.
    """
    original = np.asarray(labels, dtype=int)
    crisis_probs = np.asarray(crisis_probs, dtype=float)
    n = len(original)
    if n == 0:
        return original.copy()
    if len(crisis_probs) != n:
        raise ValueError("labels and crisis_probs must have equal length")

    out = original.copy()
    for t in range(1, n):
        for island_len in range(1, max_island_len + 1):
            start = t - island_len
            if start < 1:
                continue
            island_reg = int(original[start])
            if np.any(original[start:t] != island_reg):
                continue
            left = int(out[start - 1])
            right = int(original[t])
            if left != right or island_reg == left:
                continue
            if island_reg == CRISIS_REGIME_ID:
                if float(crisis_probs[start:t].max()) >= crisis_keep_prob:
                    continue
            out[start:t] = left
    return out


def fit_static_gmm(
    features: pd.DataFrame,
    k: int,
    *,
    random_state: int = 42,
    smooth_labels: Optional[bool] = None,
) -> Tuple[GaussianMixture, StandardScaler, np.ndarray, dict[int, int]]:
    """
    Fit in-sample GMM, label with template Hungarian, return hard labels.

    For K=5 applies trailing 3-month mode smoothing on labels (causal window;
    ``center=False`` — no future months in the rolling window). K=3 and K=4 stay
    unsmoothed by default.
    """
    if smooth_labels is None:
        smooth_labels = k == 5

    cfg = _gmm_config(k)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    gmm = GaussianMixture(
        n_components=k,
        random_state=random_state,
        **cfg,
    )
    gmm.fit(X)

    mapping = map_clusters(gmm, features.columns, k=k)
    raw_labels = gmm.predict(X)
    labels = relabel_states(raw_labels, mapping)

    if smooth_labels:
        s = pd.Series(labels, index=features.index)
        labels = (
            s.rolling(window=3, center=False, min_periods=1)
            .apply(lambda x: pd.Series(x).mode().iloc[0])
            .astype(int)
            .values
        )

    return gmm, scaler, labels, mapping


def run_walk_forward(
    features: pd.DataFrame,
    k: int,
    *,
    test_start: str = "1990-01-31",
    map_clusters_monthly: Optional[bool] = None,
    use_dwell_hysteresis: Optional[bool] = None,
    use_island_suppression: Optional[bool] = None,
    calm_dwell_months: int = 2,
    crisis_enter_months: int = 1,
    crisis_exit_months: int = 1,
    crisis_enter_weak_months: int = 2,
    crisis_strong_prob: float = 0.32,
    crisis_keep_prob: float = 0.28,
    random_state: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Expanding-window GMM from ``test_start`` with:
    - expanding StandardScaler (not EWZ)
    - temporal Hungarian in raw space every month
    - K=3/K=4: monthly economic relabeling, no dwell/island cleanup
    - K=5: January economic relabeling, asymmetric dwell (no island suppression
      by default — see ``use_island_suppression``)
    - Layer 1: moderate EMA (K=3/K=4: span 3 / 60% raw; K=5: span 9 / 33% raw)

    Island suppression is off by default for all ``k`` so ``Regime`` has no
    retroactive relabeling or confirmation lag. Pass ``use_island_suppression=True``
    only for non-trading display experiments.
    """
    dates = features.index
    n_obs = len(dates)
    start_idx = int(np.searchsorted(dates.values, pd.Timestamp(test_start).to_datetime64(), side="left"))
    if start_idx >= n_obs:
        raise ValueError("test_start is after end of sample")

    cfg = _gmm_config(k)
    ema_span, prob_raw_weight = _wf_smoothing_config(k)
    if map_clusters_monthly is None:
        map_clusters_monthly = k in {3, 4}
    if use_dwell_hysteresis is None:
        use_dwell_hysteresis = k == 5
    if use_island_suppression is None:
        use_island_suppression = False

    gmm = GaussianMixture(
        n_components=k,
        warm_start=False,
        random_state=random_state,
        **cfg,
    )

    track_w = tracker_feature_weights(features.columns)
    prev_means_raw: Optional[np.ndarray] = None
    canon_to_regime: Optional[dict[int, int]] = None

    prob_rows: list[np.ndarray] = []
    row_dates: list[pd.Timestamp] = []

    iterator = range(start_idx, n_obs)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Walk-forward GMM K={k}")

    for t_idx in iterator:
        date_t = dates[t_idx]
        X_train = features.iloc[:t_idx].values
        X_curr = features.iloc[t_idx : t_idx + 1].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_curr_s = scaler.transform(X_curr)

        gmm.fit(X_train_s)
        means_raw = scaler.inverse_transform(gmm.means_)

        if prev_means_raw is None:
            canon_to_sklearn = np.arange(k, dtype=int)
        else:
            canon_to_sklearn = hungarian_align_means(means_raw, prev_means_raw, track_w)

        prev_means_raw = means_raw[canon_to_sklearn].copy()

        do_map = map_clusters_monthly or t_idx == start_idx or date_t.month == 1
        if do_map:
            skl_to_reg = map_clusters(gmm, features.columns, k=k)
            canon_to_regime = {
                j: int(skl_to_reg[int(canon_to_sklearn[j])]) for j in range(k)
            }
        assert canon_to_regime is not None

        p_skl = gmm.predict_proba(X_curr_s)[0]
        p_canon = np.array([float(p_skl[int(canon_to_sklearn[j])]) for j in range(k)])
        anchored = np.zeros(k, dtype=float)
        for j in range(k):
            anchored[canon_to_regime[j]] += p_canon[j]

        row_dates.append(date_t)
        prob_rows.append(anchored)

    prob_mat = np.stack(prob_rows, axis=0)
    prob_cols = [f"Prob_Regime{i}" for i in range(k)]
    prob_df = pd.DataFrame(prob_mat, index=pd.DatetimeIndex(row_dates), columns=prob_cols)

    ema = prob_df.ewm(span=ema_span, adjust=False).mean()
    w = float(np.clip(prob_raw_weight, 0.0, 1.0))
    combo = w * prob_df + (1.0 - w) * ema
    combo = combo.div(combo.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    results = combo.copy()
    candidates = combo.values.argmax(axis=1).astype(int)
    results["Regime_Candidate"] = candidates

    crisis_probs = combo[f"Prob_Regime{CRISIS_REGIME_ID}"].values

    if use_dwell_hysteresis:
        labels = apply_dwell_hysteresis(
            candidates,
            crisis_probs,
            k=k,
            calm_dwell_months=calm_dwell_months,
            crisis_enter_months=crisis_enter_months,
            crisis_exit_months=crisis_exit_months,
            crisis_enter_weak_months=crisis_enter_weak_months,
            crisis_strong_prob=crisis_strong_prob,
        )
    else:
        labels = candidates.copy()

    if use_island_suppression:
        labels = suppress_island_flickers(
            labels,
            crisis_probs,
            crisis_keep_prob=crisis_keep_prob,
        )

    results["Regime"] = labels
    results["Regime_Name"] = [regime_names_from_k(k)[r] for r in results["Regime"]]

    sanity_check(results["Regime"].values, results.index, k)
    return results


def regime_names_from_k(k: int) -> dict[int, str]:
    from scripts.regime_labeling import get_spec

    return get_spec(k).names


def static_regime_table(
    features: pd.DataFrame, labels: np.ndarray, k: int
) -> pd.DataFrame:
    from scripts.regime_labeling import get_spec

    spec = get_spec(k)
    rows = []
    for reg_id in range(k):
        mask = labels == reg_id
        sub = features.loc[mask]
        rows.append(
            {
                "Regime_ID": reg_id,
                "Regime_Name": spec.names[reg_id],
                "Months": int(mask.sum()),
                "Mean_SPXT": sub["SPXT"].mean() if mask.any() else np.nan,
                "Std_SPXT": sub["SPXT"].std() if mask.any() else np.nan,
                "Mean_VIX": sub["VIX"].mean() if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
