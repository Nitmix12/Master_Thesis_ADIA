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


def load_backtest_panel(
    *,
    test_start: str = DEFAULT_TEST_START,
    features_path: Path | None = None,
) -> pd.DataFrame:
    """
    Monthly asset returns for backtesting (aligned on ``features.csv`` index).

    Columns: SPXT (equity), LUATTRUU (treasuries), BCOMTR (commodities).
    """
    features = load_features() if features_path is None else pd.read_csv(
        features_path, index_col=0, parse_dates=True
    ).sort_index()
    panel = features[[EQUITY_COL, SAFE_HAVEN_COL, COMMODITY_COL]].apply(
        pd.to_numeric, errors="coerce"
    )
    panel = panel.loc[panel.index >= pd.Timestamp(test_start)].dropna(how="any")
    return panel


def load_walk_forward_predictions(
    k: int,
    *,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    """Load ``walk_forward_k4.csv`` or ``walk_forward_k5.csv``."""
    if k not in (4, 5):
        raise ValueError("k must be 4 or 5")
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
