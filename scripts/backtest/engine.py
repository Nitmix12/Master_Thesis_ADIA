"""Monthly backtest engine, dollar accounting, metrics, and plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.backtest.loaders import (
    COMMODITY_COL,
    EQUITY_COL,
    SAFE_HAVEN_COL,
    load_backtest_panel,
)
from scripts.backtest.strategies import StrategyWeights

DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_MONTHLY_CONTRIBUTION = 10_000.0
MONTHS_PER_YEAR = 12


def annualize_return(monthly_returns: pd.Series) -> float:
    r = pd.to_numeric(monthly_returns, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    years = len(r) / MONTHS_PER_YEAR
    if years <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def annualize_vol(monthly_returns: pd.Series) -> float:
    r = pd.to_numeric(monthly_returns, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def _max_underwater_months(nav: pd.Series) -> int:
    if nav.empty:
        return 0
    running_max = nav.cummax()
    underwater = nav < running_max
    max_dur = 0
    dur = 0
    for flag in underwater.astype(bool):
        if flag:
            dur += 1
            max_dur = max(max_dur, dur)
        else:
            dur = 0
    return int(max_dur)


def run_single_asset_backtest(
    asset_returns: pd.Series,
    target_weight: pd.Series,
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    ret = pd.to_numeric(asset_returns, errors="coerce").dropna().sort_index()
    w = pd.to_numeric(target_weight, errors="coerce").reindex(ret.index).ffill().fillna(0.0)
    w = w.clip(0.0, 1.0)
    w_exec = w.shift(1).fillna(0.0)
    cash_w = 1.0 - w_exec
    gross_ret = (w_exec * ret) + (cash_w * 0.0)
    turnover = (w_exec - w_exec.shift(1)).abs().fillna(w_exec.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_ret = gross_ret - cost
    nav_net = (1.0 + net_ret).cumprod()
    drawdown = (nav_net / nav_net.cummax()) - 1.0
    return pd.DataFrame(
        {
            "Net_Return": net_ret,
            "NAV_Net": nav_net,
            "Drawdown_Net": drawdown,
            "Turnover": turnover,
            "Exec_Weight_Equity": w_exec,
        },
        index=ret.index,
    )


def run_two_asset_backtest(
    equity_returns: pd.Series,
    safe_haven_returns: pd.Series,
    equity_target_weight: pd.Series,
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    eq = pd.to_numeric(equity_returns, errors="coerce").dropna().sort_index()
    sh = pd.to_numeric(safe_haven_returns, errors="coerce").dropna().sort_index()
    idx = eq.index.intersection(sh.index)
    eq, sh = eq.reindex(idx), sh.reindex(idx)
    w_eq = pd.to_numeric(equity_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0)
    w_eq = w_eq.clip(0.0, 1.0)
    w_eq_exec = w_eq.shift(1).fillna(0.0)
    w_sh_exec = 1.0 - w_eq_exec
    gross_ret = (w_eq_exec * eq) + (w_sh_exec * sh)
    turnover = (2.0 * (w_eq_exec - w_eq_exec.shift(1)).abs()).fillna(2.0 * w_eq_exec.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_ret = gross_ret - cost
    nav_net = (1.0 + net_ret).cumprod()
    drawdown = (nav_net / nav_net.cummax()) - 1.0
    return pd.DataFrame(
        {
            "Net_Return": net_ret,
            "NAV_Net": nav_net,
            "Drawdown_Net": drawdown,
            "Turnover": turnover,
            "Exec_Weight_Equity": w_eq_exec,
        },
        index=idx,
    )


def run_three_asset_backtest(
    equity_returns: pd.Series,
    safe_haven_returns: pd.Series,
    commodity_returns: pd.Series,
    equity_target_weight: pd.Series,
    safe_haven_target_weight: pd.Series,
    commodity_target_weight: pd.Series,
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    eq = pd.to_numeric(equity_returns, errors="coerce").dropna().sort_index()
    sh = pd.to_numeric(safe_haven_returns, errors="coerce").dropna().sort_index()
    cm = pd.to_numeric(commodity_returns, errors="coerce").dropna().sort_index()
    idx = eq.index.intersection(sh.index).intersection(cm.index)
    eq, sh, cm = eq.reindex(idx), sh.reindex(idx), cm.reindex(idx)

    w_eq = pd.to_numeric(equity_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
    w_sh = pd.to_numeric(safe_haven_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
    w_cm = pd.to_numeric(commodity_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)

    target = pd.concat([w_eq, w_sh, w_cm], axis=1)
    target.columns = ["eq", "sh", "cm"]
    target = target.div(target.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    w_exec = target.shift(1).fillna(0.0)
    gross_ret = (w_exec["eq"] * eq) + (w_exec["sh"] * sh) + (w_exec["cm"] * cm)
    turnover = (w_exec - w_exec.shift(1)).abs().sum(axis=1).fillna(w_exec.abs().sum(axis=1))
    cost = turnover * (transaction_cost_bps / 10_000.0)
    net_ret = gross_ret - cost
    nav_net = (1.0 + net_ret).cumprod()
    drawdown = (nav_net / nav_net.cummax()) - 1.0
    return pd.DataFrame(
        {
            "Net_Return": net_ret,
            "NAV_Net": nav_net,
            "Drawdown_Net": drawdown,
            "Turnover": turnover,
        },
        index=idx,
    )


def run_strategy_backtest(
    weights: StrategyWeights,
    returns_panel: pd.DataFrame,
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    eq = returns_panel[EQUITY_COL]
    sh = returns_panel[SAFE_HAVEN_COL]
    cm = returns_panel[COMMODITY_COL]

    if weights.kind == "single":
        if weights.equity is None:
            raise ValueError("single-asset strategy requires equity weights")
        return run_single_asset_backtest(
            eq, weights.equity, transaction_cost_bps=transaction_cost_bps
        )
    if weights.kind == "two":
        if weights.equity is None:
            raise ValueError("two-asset strategy requires equity weights")
        return run_two_asset_backtest(
            eq, sh, weights.equity, transaction_cost_bps=transaction_cost_bps
        )
    if weights.kind == "three":
        if weights.equity is None or weights.safe_haven is None or weights.commodity is None:
            raise ValueError("three-asset strategy requires equity, safe_haven, commodity weights")
        return run_three_asset_backtest(
            eq,
            sh,
            cm,
            weights.equity,
            weights.safe_haven,
            weights.commodity,
            transaction_cost_bps=transaction_cost_bps,
        )
    raise ValueError(f"Unknown strategy kind: {weights.kind}")


def dollar_portfolio_curve(
    net_returns: pd.Series,
    *,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
) -> pd.Series:
    """Portfolio value with a fixed contribution invested each month before return."""
    values: list[float] = []
    wealth = 0.0
    for r in pd.to_numeric(net_returns, errors="coerce").fillna(0.0):
        wealth = (wealth + float(monthly_contribution)) * (1.0 + float(r))
        values.append(wealth)
    return pd.Series(values, index=net_returns.index, name="Portfolio_USD")


def compute_metrics(
    backtest: pd.DataFrame,
    *,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
    label: str = "",
) -> pd.Series:
    r = pd.to_numeric(backtest["Net_Return"], errors="coerce").dropna()
    nav = pd.to_numeric(backtest["NAV_Net"], errors="coerce").dropna()
    dd = pd.to_numeric(backtest["Drawdown_Net"], errors="coerce").dropna()
    port = dollar_portfolio_curve(r, monthly_contribution=monthly_contribution)

    months = len(r)
    total_invested = months * float(monthly_contribution)
    final_value = float(port.iloc[-1]) if len(port) else float("nan")
    profit = final_value - total_invested
    multiple = final_value / total_invested if total_invested > 0 else float("nan")

    cagr = annualize_return(r)
    vol = annualize_vol(r)
    sharpe = float(cagr / vol) if np.isfinite(vol) and vol > 0 else float("nan")
    max_dd = float(dd.min()) if len(dd) else float("nan")
    calmar = float(cagr / abs(max_dd)) if max_dd not in (0.0, np.nan) and np.isfinite(max_dd) else float("nan")

    downside = r.copy()
    downside[downside > 0] = 0.0
    down_vol = float(downside.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if len(downside) > 1 else float("nan")
    sortino = float(cagr / down_vol) if np.isfinite(down_vol) and down_vol > 0 else float("nan")

    return pd.Series(
        {
            "Label": label,
            "Months": months,
            "Total_Invested_USD": total_invested,
            "Final_Portfolio_USD": final_value,
            "Profit_USD": profit,
            "Multiple": multiple,
            "CAGR": cagr,
            "Annual_Vol": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max_Drawdown": max_dd,
            "Calmar": calmar,
            "Win_Rate": float((r > 0).mean()) if len(r) else float("nan"),
            "Avg_Monthly_Turnover": float(backtest["Turnover"].mean()) if "Turnover" in backtest else float("nan"),
            "Max_Underwater_Months": _max_underwater_months(nav),
            "Final_NAV_Multiple": float(nav.iloc[-1]) if len(nav) else float("nan"),
        }
    )


def plot_strategy_report(
    results_by_label: dict[str, pd.DataFrame],
    metrics_table: pd.DataFrame,
    *,
    title: str,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
) -> plt.Figure:
    """Portfolio USD, drawdown, and metrics table for one or more K / variants."""
    n = len(results_by_label)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_val, ax_dd = axes

    for label, bt in results_by_label.items():
        port = dollar_portfolio_curve(bt["Net_Return"], monthly_contribution=monthly_contribution)
        ax_val.plot(port.index, port.values, linewidth=1.4, label=label)
        ax_dd.plot(bt.index, bt["Drawdown_Net"], linewidth=1.0, label=label)

    ax_val.set_ylabel("Portfolio value (USD)")
    ax_val.set_title(title)
    ax_val.legend(loc="upper left", fontsize=9)
    ax_val.grid(True, alpha=0.3)

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(loc="lower left", fontsize=9)
    ax_dd.grid(True, alpha=0.3)

    # Sharpe annotation from metrics table
    if not metrics_table.empty and "Sharpe" in metrics_table.columns:
        lines = []
        for idx, row in metrics_table.iterrows():
            sh = row.get("Sharpe", np.nan)
            if np.isfinite(sh):
                lines.append(f"{idx}: Sharpe={sh:.2f}")
        if lines:
            ax_val.text(
                0.99,
                0.02,
                "\n".join(lines),
                transform=ax_val.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    plt.tight_layout()

    display_cols = [
        "Months",
        "Total_Invested_USD",
        "Final_Portfolio_USD",
        "Profit_USD",
        "Multiple",
        "CAGR",
        "Annual_Vol",
        "Sharpe",
        "Sortino",
        "Max_Drawdown",
        "Calmar",
        "Win_Rate",
        "Avg_Monthly_Turnover",
    ]
    existing = [c for c in display_cols if c in metrics_table.columns]
    print(metrics_table[existing].to_string(float_format=lambda x: f"{x:,.4f}" if abs(x) < 100 else f"{x:,.0f}"))

    return fig


def run_strategy_comparison_cell(
    strategy_key: str,
    *,
    soft: bool | None = None,
    returns_panel: pd.DataFrame | None = None,
    signals_k4: Any = None,
    signals_k5: Any = None,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
    title: str | None = None,
) -> pd.DataFrame:
    """
    Run one strategy for K=4 and K=5 (and buy-and-hold once), plot, return metrics table.
    """
    from scripts.backtest.signals import load_walk_forward_signals
    from scripts.backtest.strategies import STRATEGY_BUILDERS

    if returns_panel is None:
        returns_panel = load_backtest_panel()
    if signals_k4 is None:
        signals_k4 = load_walk_forward_signals(4)
    if signals_k5 is None:
        signals_k5 = load_walk_forward_signals(5)

    builder = STRATEGY_BUILDERS[strategy_key]
    results: dict[str, pd.DataFrame] = {}
    metrics_rows: list[pd.Series] = []

    def _run(k: int, sig: Any, tag: str) -> None:
        if strategy_key == "buy_and_hold":
            w = builder(sig, soft=False)
        else:
            w = builder(sig, soft=bool(soft))
        bt = run_strategy_backtest(w, returns_panel)
        results[tag] = bt
        metrics_rows.append(compute_metrics(bt, monthly_contribution=monthly_contribution, label=tag))

    if strategy_key == "buy_and_hold":
        _run(4, signals_k4, "K=4 / K=5 (same)")
    else:
        _run(4, signals_k4, f"K=4 {'soft' if soft else 'hard'}")
        _run(5, signals_k5, f"K=5 {'soft' if soft else 'hard'}")

    metrics_df = pd.DataFrame(metrics_rows).set_index("Label")
    mode = "soft" if soft else "hard" if soft is not None else ""
    plot_title = title or f"{strategy_key.replace('_', ' ').title()} ({mode})".strip()
    plot_strategy_report(results, metrics_df, title=plot_title, monthly_contribution=monthly_contribution)
    return metrics_df
