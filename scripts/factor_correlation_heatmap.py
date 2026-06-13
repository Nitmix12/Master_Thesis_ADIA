"""
Correlation heatmap of the 17 Bloomberg macro/style factors (slide 4).

Loads ``data/features.csv``, orders factors by economic block, and saves a
heatmap to ``outputs/figures/factor_correlation_heatmap.png``.

Run from anywhere::

    python pmr_paper/scripts/factor_correlation_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Factor blocks for slide 4 (tickers match data_preparation.FEATURE_COLUMNS).
FACTOR_GROUPS: list[tuple[str, list[str]]] = [
    ("Equities", ["SPXT", "MXEF"]),
    ("Rates / bonds", ["USGG3M", "LUATTRUU", "LF98TRUU"]),
    ("Credit", ["LUACOAS", "NEIXCTAT"]),
    ("Commodities / inflation", ["BCOMTR", "BCIT1T"]),
    ("FX / carry", ["DXY", "DBFXCARU"]),
    ("Vol", ["VIX"]),
    (
        "MSCI World style",
        ["M1WOSC", "M1WO000V", "M1WOMOM", "M1WOQU", "M1WOMVOL"],
    ),
]

GROUP_COLORS = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#8C8C8C",
]

OUTPUT_NAME = "factor_correlation_heatmap.png"


def _ordered_tickers() -> list[str]:
    tickers: list[str] = []
    for _, group in FACTOR_GROUPS:
        tickers.extend(group)
    return tickers


def plot_factor_correlation_heatmap(
    features: pd.DataFrame,
    *,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (11, 9),
    dpi: int = 200,
    show: bool = False,
) -> plt.Figure:
    """Pearson correlation heatmap of 17 factors, grouped by economic block."""
    order = _ordered_tickers()
    missing = [t for t in order if t not in features.columns]
    if missing:
        raise ValueError(f"features missing columns: {missing}")

    data = features[order]
    corr = data.corr()
    n = len(order)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(order, fontsize=9)

    # Block separators between factor groups.
    boundaries = np.cumsum([len(g) for _, g in FACTOR_GROUPS[:-1]])
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=1.5)
        ax.axvline(b - 0.5, color="white", linewidth=1.5)

    # Category color strips (left / top).
    starts = np.cumsum([0] + [len(g) for _, g in FACTOR_GROUPS])
    for i, ((label, _), color) in enumerate(zip(FACTOR_GROUPS, GROUP_COLORS)):
        lo, hi = starts[i], starts[i + 1]
        ax.add_patch(
            plt.Rectangle(
                (-1.35, lo - 0.5),
                0.25,
                hi - lo,
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (lo - 0.5, n + 0.15),
                hi - lo,
                0.25,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                facecolor=color,
                edgecolor="none",
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("Pearson correlation", fontsize=10)

    start = data.index.min().strftime("%b %Y")
    end = data.index.max().strftime("%b %Y")
    ax.set_title(
        f"17-factor correlation matrix ({start} – {end}, monthly)",
        fontsize=12,
        pad=12,
    )

    fig.subplots_adjust(left=0.14, bottom=0.22, right=0.92, top=0.92)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.paths import FIGURES_DIR, load_features

    features = load_features()
    out_path = FIGURES_DIR / OUTPUT_NAME
    plot_factor_correlation_heatmap(features, save_path=out_path, show=False)
    print(f"Saved: {out_path}")
    print(f"Sample: {features.index.min().date()} → {features.index.max().date()} "
          f"({len(features)} months, {features.shape[1]} factors)")


if __name__ == "__main__":
    main()
