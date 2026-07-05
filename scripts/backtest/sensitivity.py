"""
Sensitivity analysis for walk-forward GMM smoothing parameters (Phase 3).

Sweeps EMA half-life (x-axis) × min-dwell N (y-axis) on a
*pre-1990 only* evaluation window and computes Sharpe ratio and
win rate (hit ratio) for a fixed representative strategy.

Conversion:  half-life h  →  EWM span
    alpha  = 1 - 0.5^(1/h)
    span   = 2/alpha - 1
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.backtest.engine import (
    annualize_return,
    annualize_vol,
    run_strategy_backtest,
)
from scripts.backtest.signals import RegimeSignal
from scripts.backtest.strategies import STRATEGY_BUILDERS
from scripts.gmm_pipeline import run_walk_forward


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def half_life_to_span(h: float) -> float:
    """
    Convert EWM half-life (months) to the pandas ``span`` parameter.

    Derivation:
        The EWM decay factor: alpha = 1 - exp(-ln2 / h) = 1 - 0.5^(1/h)
        pandas span definition: alpha = 2 / (span + 1)  →  span = 2/alpha - 1
    """
    alpha = 1.0 - 0.5 ** (1.0 / float(h))
    return 2.0 / alpha - 1.0


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def run_sensitivity_grid(
    features: pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    k: int = 5,
    strategy_key: str = "bond_floor_tactical",
    half_lives: Sequence[float] = (1, 2, 3, 4, 6, 9, 12),
    dwell_ns: Sequence[int] = (1, 2, 3, 4),
    wf_test_start: str = "1975-01-31",
    eval_start: str = "1975-01-31",
    eval_end: str = "1990-12-31",
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep ``(half_life, dwell_N)`` on the pre-1990 selection window.

    Parameters
    ----------
    features       : full feature matrix (scaled inside run_walk_forward)
    returns_panel  : monthly returns for SPXT / LUATTRUU / BCOMTR
    k              : number of GMM regimes (default 5)
    strategy_key   : strategy from STRATEGY_BUILDERS (default "bond_floor_tactical")
    half_lives     : EWM half-life values to try (x-axis of heatmap)
    dwell_ns       : calm_dwell_months values to try (y-axis of heatmap)
    wf_test_start  : first date of expanding walk-forward window
    eval_start     : first date of backtest evaluation slice
    eval_end       : last date of backtest evaluation slice (must be <= 1990-12-31)
    show_progress  : pass True to show tqdm bars per run

    Returns
    -------
    sharpe_grid, winrate_grid : DataFrames indexed by dwell_N, columns = half_life
    """
    builder = STRATEGY_BUILDERS[strategy_key]

    half_lives_list = list(half_lives)
    dwell_ns_list = list(dwell_ns)

    sharpe_values = np.full((len(dwell_ns_list), len(half_lives_list)), np.nan)
    winrate_values = np.full((len(dwell_ns_list), len(half_lives_list)), np.nan)

    total = len(half_lives_list) * len(dwell_ns_list)
    done = 0

    for j, h in enumerate(half_lives_list):
        span = half_life_to_span(h)

        for i, n in enumerate(dwell_ns_list):
            done += 1
            print(f"  [{done:2d}/{total}]  half_life={h:5.1f}  dwell_N={n}", end="")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wf = run_walk_forward(
                        features,
                        k=k,
                        test_start=wf_test_start,
                        calm_dwell_months=n,
                        ema_span_override=int(round(span)),
                        show_progress=show_progress,
                    )
            except Exception as exc:
                print(f"  ← FAILED: {exc}")
                continue

            # Slice to evaluation window
            wf_eval = wf.loc[eval_start:eval_end]
            if len(wf_eval) < 12:
                print(f"  ← too few months ({len(wf_eval)}), skipped")
                continue

            # Build RegimeSignal for the eval window
            prob_cols = [c for c in wf_eval.columns if c.startswith("Prob_Regime")]
            signal = RegimeSignal(
                k=k,
                index=pd.DatetimeIndex(wf_eval.index),
                regime_id=wf_eval["Regime"].astype(int),
                regime_name=wf_eval["Regime_Name"].astype(str),
                probabilities=wf_eval[prob_cols],
            )

            weights = builder(signal, soft=False)

            # Align returns to eval window
            panel_eval = returns_panel.reindex(wf_eval.index).dropna(how="any")
            if panel_eval.empty:
                print("  ← no return data")
                continue

            bt = run_strategy_backtest(weights, panel_eval)

            net_ret = pd.to_numeric(bt["Net_Return"], errors="coerce").dropna()
            if len(net_ret) < 12:
                print(f"  ← too few return months ({len(net_ret)}), skipped")
                continue

            cagr = annualize_return(net_ret)
            vol = annualize_vol(net_ret)
            sharpe = float(cagr / vol) if np.isfinite(vol) and vol > 0 else np.nan
            win_rate = float((net_ret > 0).mean())

            sharpe_values[i, j] = round(sharpe, 4)
            winrate_values[i, j] = round(win_rate, 4)

            print(f"  Sharpe={sharpe:.3f}  WinRate={win_rate:.3f}")

    idx = pd.Index(dwell_ns_list, name="dwell_N")
    cols = pd.Index([float(h) for h in half_lives_list], name="half_life")

    sharpe_grid = pd.DataFrame(sharpe_values, index=idx, columns=cols)
    winrate_grid = pd.DataFrame(winrate_values, index=idx, columns=cols)
    return sharpe_grid, winrate_grid


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

def _draw_heatmap(
    ax: plt.Axes,
    grid: pd.DataFrame,
    title: str,
    fmt: str,
    cmap: str,
    *,
    chosen_h: float | None = None,
    chosen_N: int | None = None,
) -> None:
    """Draw a single annotated heatmap using matplotlib only (no seaborn)."""
    values = grid.values.astype(float)
    masked = np.ma.masked_invalid(values)

    im = ax.imshow(masked, aspect="auto", cmap=cmap, origin="upper")

    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            text = fmt % v if np.isfinite(v) else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(c) for c in grid.columns])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(r) for r in grid.index])

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("EWM half-life (months)", fontsize=10)
    ax.set_ylabel("Min-dwell N (months)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if chosen_h is not None and chosen_N is not None:
        col_labels = list(grid.columns)
        row_labels = list(grid.index)
        if chosen_h in col_labels and chosen_N in row_labels:
            cx = col_labels.index(chosen_h)
            cy = row_labels.index(chosen_N)
            ax.plot(
                cx,
                cy,
                marker="*",
                color="red",
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"Chosen (h={chosen_h}, N={chosen_N})",
                zorder=5,
            )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)


def plot_sensitivity_heatmaps(
    sharpe_grid: pd.DataFrame,
    winrate_grid: pd.DataFrame,
    *,
    chosen_h: float | None = None,
    chosen_N: int | None = None,
    strategy_key: str = "bond_floor_tactical",
    eval_start: str = "1975",
    eval_end: str = "1990",
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Draw Figure A (Sharpe) and Figure B (win rate) side by side.

    Marks the production default ``(chosen_h, chosen_N)`` with a red star.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    strat_label = strategy_key.replace("_", " ").title()
    _draw_heatmap(
        axes[0],
        sharpe_grid,
        f"Figure A — Sharpe ratio\n{strat_label} | {eval_start}–{eval_end}",
        "%.2f",
        "RdYlGn",
        chosen_h=chosen_h,
        chosen_N=chosen_N,
    )
    _draw_heatmap(
        axes[1],
        winrate_grid,
        f"Figure B — Win rate (hit ratio)\n{strat_label} | {eval_start}–{eval_end}",
        "%.2f",
        "RdYlGn",
        chosen_h=chosen_h,
        chosen_N=chosen_N,
    )

    plt.tight_layout()
    return fig


def production_half_life(k: int = 5) -> float:
    """
    Return the production EWM half-life for a given K, derived from the span
    used in ``_wf_smoothing_config``.

    Inverse of ``half_life_to_span``:  h = -ln2 / ln(1 - 2/(span+1))
    """
    from scripts.gmm_pipeline import _wf_smoothing_config
    span, _ = _wf_smoothing_config(k)
    alpha = 2.0 / (span + 1.0)
    return -np.log(2.0) / np.log(1.0 - alpha)
