"""Load monthly returns and walk-forward regime outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from scripts.paths import FEATURES_PATH, OUTPUT_DIR, load_features

DEFAULT_TEST_START = "1990-01-31"
EQUITY_COL = "SPXT"
SAFE_HAVEN_COL = "LUATTRUU"
COMMODITY_COL = "BCOMTR"
ALT_BOND_COL = "LF98TRUU"
EM_EQUITY_COL = "MXEF"
TIPS_COL = "BCIT1T"

# Equal-weight three-asset benchmark (investable universe for regime strategies).
EW_THREE_COLS: tuple[str, ...] = (EQUITY_COL, SAFE_HAVEN_COL, COMMODITY_COL)
EW_THREE_WEIGHT: float = 1.0 / 3.0

# Extended six-asset investable universe (total-return indices in features.csv).
CORE6_COLS: tuple[str, ...] = (
    EQUITY_COL,
    SAFE_HAVEN_COL,
    ALT_BOND_COL,
    COMMODITY_COL,
    EM_EQUITY_COL,
    TIPS_COL,
)
EW_SIX_COLS: tuple[str, ...] = CORE6_COLS
EW_SIX_WEIGHT: float = 1.0 / 6.0


def load_backtest_panel(
    *,
    test_start: str = DEFAULT_TEST_START,
    features_path: Path | None = None,
) -> pd.DataFrame:
    """
    Monthly asset returns for backtesting (aligned on ``features.csv`` index).

    Columns: SPXT (equity), LUATTRUU (treasuries), BCOMTR (commodities).
    These three assets are the investable universe for regime portfolios and for
    the EW3 benchmark (1/3 each), not an equal-weight across all 17 GMM factors.
    """
    features = load_features() if features_path is None else pd.read_csv(
        features_path, index_col=0, parse_dates=True
    ).sort_index()
    panel = features[[EQUITY_COL, SAFE_HAVEN_COL, COMMODITY_COL]].apply(
        pd.to_numeric, errors="coerce"
    )
    panel = panel.loc[panel.index >= pd.Timestamp(test_start)].dropna(how="any")
    return panel


def load_core6_backtest_panel(
    *,
    test_start: str = DEFAULT_TEST_START,
    features_path: Path | None = None,
) -> pd.DataFrame:
    """
    Monthly returns for the six-asset extended investable universe.

    Columns: SPXT, LUATTRUU, LF98TRUU, BCOMTR, MXEF, BCIT1T.
    """
    features = load_features() if features_path is None else pd.read_csv(
        features_path, index_col=0, parse_dates=True
    ).sort_index()
    panel = features[list(CORE6_COLS)].apply(pd.to_numeric, errors="coerce")
    panel = panel.loc[panel.index >= pd.Timestamp(test_start)].dropna(how="any")
    return panel


def load_walk_forward_predictions(
    k: int,
    *,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    """Load ``walk_forward_k3.csv``, ``walk_forward_k4.csv``, or ``walk_forward_k5.csv``."""
    if k not in (3, 4, 5):
        raise ValueError("k must be 3, 4, or 5")
    out_dir = outputs_dir or OUTPUT_DIR
    path = out_dir / f"walk_forward_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run notebooks/models/02_walk_forward_gmm.ipynb first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def load_regime_backtest_bundle(
    k: int,
    *,
    test_start: str = DEFAULT_TEST_START,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns
    -------
    returns_panel, walk_forward_df, aligned_regime_id
    """
    panel = load_backtest_panel(test_start=test_start)
    wf = load_walk_forward_predictions(k)
    common = panel.index.intersection(wf.index)
    panel = panel.reindex(common)
    wf = wf.reindex(common)
    regime = wf["Regime"].astype(int)
    return panel, wf, regime
