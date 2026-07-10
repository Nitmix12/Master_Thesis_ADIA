"""
Data-driven regime portfolios via static GMM + mean-variance optimization.

Training uses features from ``train_start`` through ``train_end`` only (default
1971-03-31 → 1990-12-31). Regimes are identified in 17-dimensional factor space;
optimal weights are computed on the three investable sleeves (SPXT, LUATTRUU,
BCOMTR) using each component's conditional mean and covariance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from scripts.gmm_pipeline import _gmm_config, regime_names_from_k
from scripts.paths import OUTPUT_DIR, load_features
from scripts.regime_labeling import map_clusters

EQUITY_COL = "SPXT"
SAFE_HAVEN_COL = "LUATTRUU"
COMMODITY_COL = "BCOMTR"
ALT_BOND_COL = "LF98TRUU"
EM_EQUITY_COL = "MXEF"
TIPS_COL = "BCIT1T"
INVESTABLE_COLS: tuple[str, ...] = (EQUITY_COL, SAFE_HAVEN_COL, COMMODITY_COL)
CORE6_COLS: tuple[str, ...] = (
    EQUITY_COL,
    SAFE_HAVEN_COL,
    ALT_BOND_COL,
    COMMODITY_COL,
    EM_EQUITY_COL,
    TIPS_COL,
)
PORTFOLIO_VARIANTS: dict[str, tuple[str, ...]] = {
    "default": INVESTABLE_COLS,
    "core6": CORE6_COLS,
}
DEFAULT_TRAIN_START = "1971-03-31"
DEFAULT_TRAIN_END = "1990-12-31"
DEFAULT_RIDGE = 1e-5


def fit_training_gmm(
    features: pd.DataFrame,
    k: int,
    *,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    random_state: int = 42,
) -> tuple[GaussianMixture, StandardScaler, dict[int, int], pd.Index]:
    """
    Fit a static K-regime GMM on the training window only.

    Returns ``(gmm, scaler, sklearn_to_regime_mapping, feature_columns)``.
    """
    train = features.loc[train_start:train_end].dropna(how="any")
    if len(train) < k * 12:
        raise ValueError(f"Training sample too short for K={k}: {len(train)} months")

    cfg = _gmm_config(k)
    scaler = StandardScaler()
    X = scaler.fit_transform(train.values)
    feature_cols = train.columns

    gmm = GaussianMixture(
        n_components=k,
        random_state=random_state,
        **cfg,
    )
    gmm.fit(X)
    mapping = map_clusters(gmm, feature_cols, k=k)
    return gmm, scaler, mapping, feature_cols


def _component_covariance(gmm: GaussianMixture, skl_idx: int) -> np.ndarray:
    """Full covariance matrix for one sklearn component."""
    n_features = gmm.means_.shape[1]
    if gmm.covariance_type == "full":
        return np.asarray(gmm.covariances_[skl_idx], dtype=float)
    if gmm.covariance_type == "diag":
        return np.diag(np.asarray(gmm.covariances_[skl_idx], dtype=float))
    if gmm.covariance_type == "tied":
        return np.asarray(gmm.covariances_, dtype=float)
    if gmm.covariance_type == "spherical":
        return np.eye(n_features) * float(gmm.covariances_[skl_idx])
    raise ValueError(f"Unsupported covariance_type: {gmm.covariance_type}")


def _unscale_mean_cov(
    scaler: StandardScaler,
    mu_scaled: np.ndarray,
    cov_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map GMM parameters from scaled space back to original feature units."""
    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    mu_raw = mu_scaled * scale + mean
    cov_raw = (scale[:, None] * cov_scaled) * scale[None, :]
    return mu_raw, cov_raw


def _subset_investable(
    mu_raw: np.ndarray,
    cov_raw: np.ndarray,
    feature_cols: pd.Index,
    investable_cols: tuple[str, ...] = INVESTABLE_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract investable-asset mean vector and covariance sub-matrix."""
    idx = [int(feature_cols.get_loc(c)) for c in investable_cols]
    mu_sub = mu_raw[idx]
    cov_sub = cov_raw[np.ix_(idx, idx)]
    return mu_sub, cov_sub


def optimal_long_only_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """
    Long-only max-Sharpe portfolio (risk-free ≈ 0):

        w* = argmax  w'μ / sqrt(w'Σw)   s.t.  w ≥ 0,  Σw = 1
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    sigma_reg = sigma + float(ridge) * np.eye(n)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(mu @ w)
        vol = float(np.sqrt(w @ sigma_reg @ w))
        if vol < 1e-12:
            return 1e6
        return -ret / vol

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        w = np.clip(result.x, 0.0, 1.0)
        total = w.sum()
        return w / total if total > 0 else np.ones(n) / n
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


def regime_mv_weights(
    gmm: GaussianMixture,
    scaler: StandardScaler,
    feature_cols: pd.Index,
    mapping: dict[int, int],
    regime_id: int,
    *,
    investable_cols: tuple[str, ...] = INVESTABLE_COLS,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """MV-optimal long-only weights for one economic regime on investable assets."""
    skl_to_reg = {int(raw): int(reg) for raw, reg in mapping.items()}
    reg_to_skl = {reg: raw for raw, reg in skl_to_reg.items()}
    if regime_id not in reg_to_skl:
        raise ValueError(f"Regime {regime_id} not in mapping {skl_to_reg}")

    skl_idx = int(reg_to_skl[regime_id])
    mu_scaled = np.asarray(gmm.means_[skl_idx], dtype=float)
    cov_scaled = _component_covariance(gmm, skl_idx)
    mu_raw, cov_raw = _unscale_mean_cov(scaler, mu_scaled, cov_scaled)
    mu_sub, cov_sub = _subset_investable(mu_raw, cov_raw, feature_cols, investable_cols)
    return optimal_long_only_weights(mu_sub, cov_sub, ridge=ridge)


def build_regime_portfolios(
    features: pd.DataFrame,
    k: int,
    *,
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    random_state: int = 42,
    ridge: float = DEFAULT_RIDGE,
    investable_cols: tuple[str, ...] = INVESTABLE_COLS,
) -> pd.DataFrame:
    """
    Fit training GMM and return a table of MV-optimal weights per regime.

    Weight columns follow ``investable_cols`` (``w_<TICKER>`` per asset).
    """
    gmm, scaler, mapping, feature_cols = fit_training_gmm(
        features,
        k,
        train_start=train_start,
        train_end=train_end,
        random_state=random_state,
    )
    names = regime_names_from_k(k)
    rows: list[dict] = []
    for regime_id in range(k):
        weights = regime_mv_weights(
            gmm,
            scaler,
            feature_cols,
            mapping,
            regime_id,
            investable_cols=investable_cols,
            ridge=ridge,
        )
        row: dict = {
            "regime_id": regime_id,
            "regime_name": names[regime_id],
        }
        for col, w in zip(investable_cols, weights):
            row[f"w_{col}"] = float(w)
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_output_path(
    k: int,
    *,
    variant: str = "default",
    outputs_dir: Path | None = None,
) -> Path:
    out = outputs_dir or OUTPUT_DIR
    if variant == "default":
        return out / f"data_driven_portfolios_k{k}.csv"
    return out / f"data_driven_portfolios_k{k}_{variant}.csv"


def save_regime_portfolios(
    table: pd.DataFrame,
    k: int,
    *,
    variant: str = "default",
    outputs_dir: Path | None = None,
) -> Path:
    """Persist frozen regime portfolio weights."""
    path = portfolio_output_path(k, variant=variant, outputs_dir=outputs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_regime_portfolios(
    k: int,
    *,
    variant: str = "default",
    investable_cols: tuple[str, ...] | None = None,
    outputs_dir: Path | None = None,
) -> Dict[int, np.ndarray]:
    """
    Load frozen portfolios as ``{regime_id: weight_vector}``.

    Raises ``FileNotFoundError`` if the CSV has not been generated yet.
    """
    if investable_cols is None:
        investable_cols = PORTFOLIO_VARIANTS.get(variant, INVESTABLE_COLS)
    path = portfolio_output_path(k, variant=variant, outputs_dir=outputs_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run notebooks/backtest/03_data_driven_optimal_weights.ipynb first."
        )
    df = pd.read_csv(path)
    portfolios: Dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        rid = int(row["regime_id"])
        portfolios[rid] = np.array(
            [float(row[f"w_{col}"]) for col in investable_cols],
            dtype=float,
        )
    return portfolios


def train_and_save_all(
    features: pd.DataFrame | None = None,
    *,
    ks: tuple[int, ...] = (3, 4, 5),
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    outputs_dir: Path | None = None,
    variant: str = "default",
    investable_cols: tuple[str, ...] | None = None,
) -> dict[int, pd.DataFrame]:
    """Fit and save data-driven portfolios for K=3, K=4, and K=5."""
    if features is None:
        features = load_features()
    if investable_cols is None:
        investable_cols = PORTFOLIO_VARIANTS.get(variant, INVESTABLE_COLS)
    tables: dict[int, pd.DataFrame] = {}
    for k in ks:
        table = build_regime_portfolios(
            features,
            k,
            train_start=train_start,
            train_end=train_end,
            investable_cols=investable_cols,
        )
        save_regime_portfolios(table, k, variant=variant, outputs_dir=outputs_dir)
        tables[k] = table
    return tables


def portfolios_to_json_serializable(portfolios: Dict[int, np.ndarray]) -> dict:
    """JSON-friendly view of regime portfolios."""
    return {str(k): v.round(6).tolist() for k, v in portfolios.items()}


def save_portfolios_json(
    portfolios: Dict[int, np.ndarray],
    k: int,
    *,
    outputs_dir: Path | None = None,
) -> Path:
    path = (outputs_dir or OUTPUT_DIR) / f"data_driven_portfolios_k{k}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portfolios_to_json_serializable(portfolios), indent=2))
    return path
