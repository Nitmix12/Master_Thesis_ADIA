"""
Build the 17-factor monthly feature matrix for Botte & Bao-style GMM regime models.

Reads Bloomberg Excel files from ``data/raw/``, constructs return/proxy series,
and writes ``data/features.csv``.

Run from anywhere::

    python pmr_paper/scripts/data_preparation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "features.csv"

# Accept either filename for pre-1990 daily SPX (VIX proxy).
SPX_DAILY_CANDIDATES = ("SPX_daily.xlsx", "SPX_Daily_1971_1990.xlsx")

FEATURE_COLUMNS = (
    "SPXT",
    "LUATTRUU",
    "LF98TRUU",
    "BCOMTR",
    "MXEF",
    "BCIT1T",
    "DXY",
    "VIX",
    "NEIXCTAT",
    "M1WOSC",
    "M1WO000V",
    "M1WOMOM",
    "M1WOQU",
    "M1WOMVOL",
    "USGG3M",
    "DBFXCARU",
    "LUACOAS",
)


def load_and_clean_excel(filename: str, *, is_daily: bool = False) -> pd.Series:
    """Load one Bloomberg Excel file; return a numeric series indexed by date."""
    filepath = RAW_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filename} not found.")
        return pd.Series(dtype=float)

    df = pd.read_excel(filepath)
    if df.empty:
        print(f"Warning: {filename} is empty.")
        return pd.Series(dtype=float)

    df.columns = [str(col).strip() for col in df.columns]
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        print(f"Warning: {filename} has no numeric columns.")
        return pd.Series(dtype=float)

    series = numeric_df.iloc[:, 0].sort_index()
    if is_daily:
        return series
    return series.resample("ME").last().sort_index()


def load_spx_daily() -> pd.Series:
    for name in SPX_DAILY_CANDIDATES:
        series = load_and_clean_excel(name, is_daily=True)
        if not series.empty:
            return series
    return pd.Series(dtype=float)


def splice_returns(primary: pd.Series, proxy: pd.Series, scale_factor: float = 1.0) -> pd.Series:
    """Back-fill missing early history in ``primary`` using scaled proxy returns."""
    primary_ret = primary.pct_change().dropna()
    proxy_ret = proxy.pct_change().dropna() * scale_factor
    return primary_ret.combine_first(proxy_ret)


def build_features() -> pd.DataFrame:
    print(f"Looking for Excel files in: {RAW_DIR}")
    print(f"Will save CSV to: {OUTPUT_FILE}\n")
    print("Loading Excel files...")

    spxt = load_and_clean_excel("SPXT.xlsx")
    spx_price = load_and_clean_excel("SPX.xlsx")
    usgg10yr = load_and_clean_excel("USGG10YR.xlsx")
    usgg3m = load_and_clean_excel("USGG3M.xlsx")

    luattruu = load_and_clean_excel("LUATTRUU.xlsx")
    lf98truu = load_and_clean_excel("LF98TRUU.xlsx")
    luactruu = load_and_clean_excel("LUACTRUU.xlsx")
    bcomtr = load_and_clean_excel("BCOMTR.xlsx")
    mxef = load_and_clean_excel("MXEF.xlsx")
    mxwo = load_and_clean_excel("MXWO.xlsx")
    bcit1t = load_and_clean_excel("BCIT1T.xlsx")
    cpi_yoy = load_and_clean_excel("CPI_YOY.xlsx")
    dxy = load_and_clean_excel("DXY.xlsx")

    vix = load_and_clean_excel("VIX.xlsx")
    spx_daily = load_spx_daily()

    m1wosc = load_and_clean_excel("M1WOSC.xlsx")
    ru20intr = load_and_clean_excel("RU20INTR.xlsx")
    m1wo000v = load_and_clean_excel("M1WO000V.xlsx")
    rlv = load_and_clean_excel("RLV.xlsx")
    m1womom = load_and_clean_excel("M1WOMOM.xlsx")
    m1woqu = load_and_clean_excel("M1WOQU.xlsx")
    m1womvol = load_and_clean_excel("M1WOMVOL.xlsx")

    neixctat = load_and_clean_excel("NEIXCTAT.xlsx")
    dbfxcaru = load_and_clean_excel("DBFXCARU.xlsx")

    luacoas = load_and_clean_excel("LUACOAS.xlsx")
    moodcbaa = load_and_clean_excel("MOODCBAA.xlsx")
    moodcaaa = load_and_clean_excel("MOODCAAA.xlsx")

    print("Building custom proxies (VIX, trend, credit spread)...")

    spx_vol_monthly = pd.Series(dtype=float)
    if not spx_daily.empty:
        spx_daily_ret = spx_daily.pct_change().dropna()
        realized = spx_daily_ret.rolling(window=30).std() * np.sqrt(252) * 100.0
        spx_vol_monthly = realized.resample("ME").last().sort_index()

    bcom_ret = bcomtr.pct_change()
    bond_proxy_ret = (-7.0 * usgg10yr.diff() / 100.0) + (usgg10yr.shift(1) / 1200.0)
    bcom_12m = bcomtr.pct_change(12)
    usgg10yr_12m = usgg10yr.diff(12)

    bcom_pos = pd.Series(np.where(bcom_12m > 0, 1.0, -1.0), index=bcomtr.index).shift(1).fillna(1.0)
    rate_pos = pd.Series(np.where(usgg10yr_12m > 0, 1.0, -1.0), index=usgg10yr.index).shift(1).fillna(1.0)
    synth_trend_ret = 0.5 * bcom_pos * bcom_ret + 0.5 * rate_pos * bond_proxy_ret

    if not moodcaaa.empty:
        synthetic_spread = (moodcbaa - moodcaaa).sort_index()
    else:
        print("Note: MOODCAAA.xlsx not found — LUACOAS proxy uses Baa yield level only.")
        synthetic_spread = moodcbaa.sort_index()

    print("Constructing 17 factor return series...")

    spx_ret = spx_price.pct_change()
    mxwo_ret = mxwo.pct_change()
    ru20_ret = ru20intr.pct_change()
    rlv_ret = rlv.pct_change()

    series_dict = {
        "SPXT": splice_returns(spxt, spx_price),
        "LUATTRUU": luattruu.pct_change().combine_first(bond_proxy_ret),
        "LF98TRUU": lf98truu.pct_change()
        .combine_first(luactruu.pct_change())
        .combine_first(bond_proxy_ret),
        "BCOMTR": bcom_ret,
        "MXEF": mxef.pct_change().combine_first(mxwo_ret),
        "BCIT1T": bcit1t.pct_change().combine_first(cpi_yoy.diff()),
        "DXY": dxy.pct_change(),
        "VIX": vix.combine_first(spx_vol_monthly).pct_change(),
        "NEIXCTAT": neixctat.pct_change().combine_first(synth_trend_ret),
        "M1WOSC": m1wosc.pct_change().combine_first(ru20_ret.combine_first(spx_ret)),
        "M1WO000V": m1wo000v.pct_change().combine_first(rlv_ret.combine_first(spx_ret)),
        "M1WOMOM": m1womom.pct_change().combine_first(spx_ret),
        "M1WOQU": m1woqu.pct_change().combine_first(spx_ret),
        "M1WOMVOL": m1womvol.pct_change().combine_first(0.7 * spx_ret),
        "USGG3M": usgg3m.diff(),
        "DBFXCARU": dbfxcaru.pct_change().combine_first(0.3 * spx_ret + (usgg3m / 1200.0)),
        "LUACOAS": luacoas.diff().combine_first(synthetic_spread.diff()),
    }

    features = pd.DataFrame(series_dict).sort_index()
    features = features.dropna(how="any")
    features = features.loc["1971-01-31":"2026-12-31"]
    features = features[list(FEATURE_COLUMNS)]

    return features


def main() -> None:
    if not RAW_DIR.is_dir():
        print(f"Error: raw data directory not found: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features()
    features.to_csv(OUTPUT_FILE)

    print(f"\nSuccess! Created {features.shape[0]} months × {features.shape[1]} features.")
    print(f"Date range: {features.index.min().date()} → {features.index.max().date()}")
    print(f"Columns: {list(features.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
