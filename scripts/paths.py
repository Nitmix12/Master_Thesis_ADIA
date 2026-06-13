"""Project paths and feature loading for the pmr_paper pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# pmr_paper/ (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_features() -> pd.DataFrame:
    """
    Load the 17-factor monthly matrix produced by ``data_preparation.py``.

    Values are **not** standardized (returns / yield & spread changes in raw units).
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"features.csv not found at {FEATURES_PATH}. "
            "Run: python pmr_paper/scripts/data_preparation.py"
        )

    features = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()
    return features.apply(pd.to_numeric, errors="coerce").dropna(how="any")
