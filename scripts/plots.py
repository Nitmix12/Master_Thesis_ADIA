"""Regime history plots (SPXT log cumulative + colored bands)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.regime_labeling import REGIME_COLORS, get_spec


def plot_regime_history(
    features: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    *,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
    dpi: int = 150,
    show: bool = True,
) -> plt.Figure:
    spec = get_spec(k)
    dates = features.index
    spxt_ret = features["SPXT"].astype(float)
    log_cum = np.log((1.0 + spxt_ret).cumprod())
    x = np.arange(len(dates))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, log_cum.values, color="black", linewidth=1.0, label="SPXT (log cum return)")

    labs = np.asarray(labels, dtype=int)
    i = 0
    while i < len(labs):
        r = int(labs[i])
        j = i + 1
        while j < len(labs) and int(labs[j]) == r:
            j += 1
        color = REGIME_COLORS.get(r, "gray")
        ax.axvspan(i - 0.5, j - 0.5, color=color, alpha=0.5, linewidth=0)
        i = j

    ax.set_xlim(-0.5, len(dates) - 0.5)
    step = max(len(dates) // 15, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in dates[::step]], rotation=45, ha="right")
    ax.set_ylabel("Log cumulative SPXT")

    if title is None:
        title = f"GMM regimes (K={k}) — {dates[0].year}-{dates[-1].year}"
    ax.set_title(title, fontsize=13)

    line = Line2D([0], [0], color="black", linewidth=2, label="SPXT (log cum return)")
    patches = [
        Patch(facecolor=REGIME_COLORS[i], alpha=0.5, label=spec.names[i])
        for i in range(k)
    ]
    ax.legend(handles=[line] + patches, loc="upper left", framealpha=0.9)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_walk_forward_subset(
    features: pd.DataFrame,
    wf_results: pd.DataFrame,
    k: int,
    **kwargs,
) -> plt.Figure:
    """Plot WF regimes aligned to SPXT on the walk-forward date index."""
    wf_dates = wf_results.index
    sub_features = features.reindex(wf_dates).dropna(subset=["SPXT"])
    labels = wf_results.reindex(sub_features.index)["Regime"].values
    return plot_regime_history(sub_features, labels, k, **kwargs)
