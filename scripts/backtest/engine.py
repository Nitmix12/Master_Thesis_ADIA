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
    load_ew14_backtest_panel,
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


def run_eq_sh_cash_backtest(
    equity_returns: pd.Series,
    safe_haven_returns: pd.Series,
    equity_target_weight: pd.Series,
    safe_haven_target_weight: pd.Series,
    cash_target_weight: pd.Series,
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Equity + treasuries + cash (0% return); weights sum to 1 before execution lag."""
    eq = pd.to_numeric(equity_returns, errors="coerce").dropna().sort_index()
    sh = pd.to_numeric(safe_haven_returns, errors="coerce").dropna().sort_index()
    idx = eq.index.intersection(sh.index)
    eq, sh = eq.reindex(idx), sh.reindex(idx)

    w_eq = pd.to_numeric(equity_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
    w_sh = pd.to_numeric(safe_haven_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
    w_cash = pd.to_numeric(cash_target_weight, errors="coerce").reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)

    target = pd.concat([w_eq, w_sh, w_cash], axis=1)
    target.columns = ["eq", "sh", "cash"]
    target = target.div(target.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    w_exec = target.shift(1).fillna(0.0)
    gross_ret = (w_exec["eq"] * eq) + (w_exec["sh"] * sh) + (w_exec["cash"] * 0.0)
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
            "Exec_Weight_Equity": w_exec["eq"],
            "Exec_Weight_Cash": w_exec["cash"],
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


def run_multi_asset_backtest(
    returns_panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    asset_columns: tuple[str, ...],
    *,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """N-asset backtest with weights summing to 1 before execution lag."""
    cols = list(asset_columns)
    rets = returns_panel[cols].apply(pd.to_numeric, errors="coerce").dropna(how="any").sort_index()
    idx = rets.index

    w = target_weights.reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
    w = w.reindex(columns=cols, fill_value=0.0)
    target = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    w_exec = target.shift(1).fillna(0.0)
    gross_ret = (w_exec * rets).sum(axis=1)
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
    if weights.kind == "eq_sh_cash":
        if weights.equity is None or weights.safe_haven is None or weights.cash is None:
            raise ValueError("eq_sh_cash strategy requires equity, safe_haven, cash weights")
        return run_eq_sh_cash_backtest(
            eq,
            sh,
            weights.equity,
            weights.safe_haven,
            weights.cash,
            transaction_cost_bps=transaction_cost_bps,
        )
    if weights.kind == "multi":
        if weights.multi is None or weights.asset_columns is None:
            raise ValueError("multi-asset strategy requires multi weights and asset_columns")
        return run_multi_asset_backtest(
            returns_panel,
            weights.multi,
            weights.asset_columns,
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_val, ax_dd = axes

    sharpe_by_label: dict[str, float] = {}
    if not metrics_table.empty and "Sharpe" in metrics_table.columns:
        for idx, row in metrics_table.iterrows():
            sh = row.get("Sharpe", np.nan)
            if np.isfinite(sh):
                sharpe_by_label[str(idx)] = float(sh)

    for label, bt in results_by_label.items():
        port = dollar_portfolio_curve(bt["Net_Return"], monthly_contribution=monthly_contribution)
        # Single upper-left legend: series name + Sharpe (avoids a duplicate bottom-right box).
        if label in sharpe_by_label:
            legend_label = f"{label} (Sharpe={sharpe_by_label[label]:.2f})"
        else:
            legend_label = label
        ax_val.plot(port.index, port.values, linewidth=1.4, label=legend_label)
        ax_dd.plot(bt.index, bt["Drawdown_Net"], linewidth=1.0)

    ax_val.set_ylabel("Portfolio value (USD)")
    ax_val.set_title(title)
    ax_val.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_val.grid(True, alpha=0.3)

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.3)

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


def _append_benchmark_curves(
    results: dict[str, pd.DataFrame],
    metrics_rows: list[pd.Series],
    *,
    returns_panel: pd.DataFrame,
    signal: Any,
    monthly_contribution: float,
) -> None:
    """Add B0 SPXT and B1 EW3 reference curves (regime-independent)."""
    from scripts.backtest.strategies import BENCHMARK_CURVE_LABELS, STRATEGY_BUILDERS

    for bench_key, bench_label in BENCHMARK_CURVE_LABELS.items():
        if bench_label in results:
            continue
        w = STRATEGY_BUILDERS[bench_key](signal, soft=False)
        bt = run_strategy_backtest(w, returns_panel)
        results[bench_label] = bt
        metrics_rows.append(
            compute_metrics(bt, monthly_contribution=monthly_contribution, label=bench_label)
        )


def finalize_strategy_summary(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split benchmarks vs regime strategies and add display / vs-benchmark columns.

    Returns (benchmarks, regime_strategies, full_table).
    """
    from scripts.backtest.strategies import BENCHMARK_DISPLAY_NAMES

    full = summary.copy()
    if "Strategy" in full.columns:
        full["Display"] = full["Strategy"].map(
            lambda s: BENCHMARK_DISPLAY_NAMES.get(str(s), str(s).replace("_", " "))
        )
        if "Mode" in full.columns:
            regime_mask = full["Mode"] != "baseline"
            full.loc[regime_mask, "Display"] = (
                full.loc[regime_mask, "Run"].astype(str)
                + " | "
                + full.loc[regime_mask, "Strategy"].astype(str).str.replace("_", " ")
                + " ("
                + full.loc[regime_mask, "Mode"].astype(str)
                + ")"
            )

    for col in ("Sharpe", "CAGR", "Max_Drawdown", "Final_Portfolio_USD"):
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    benchmarks = full[full["Mode"] == "baseline"].copy() if "Mode" in full.columns else full.iloc[0:0]
    regime = full[full["Mode"] != "baseline"].copy() if "Mode" in full.columns else full.copy()

    if not benchmarks.empty and "Sharpe" in benchmarks.columns:
        b0 = benchmarks.loc[benchmarks["Strategy"] == "buy_and_hold", "Sharpe"]
        b1 = benchmarks.loc[benchmarks["Strategy"] == "buy_and_hold_ew_three", "Sharpe"]
        if len(b0):
            regime["Sharpe_minus_B0"] = regime["Sharpe"] - float(b0.iloc[0])
        if len(b1):
            regime["Sharpe_minus_B1"] = regime["Sharpe"] - float(b1.iloc[0])

    return benchmarks, regime, full


def run_strategy_comparison_cell(
    strategy_key: str,
    *,
    soft: bool | None = None,
    returns_panel: pd.DataFrame | None = None,
    signals_k3: Any = None,
    signals_k4: Any = None,
    signals_k5: Any = None,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
    title: str | None = None,
    include_k3: bool = True,
) -> pd.DataFrame:
    """
    Run one strategy for K=3, K=4, and K=5 (benchmarks once), plot, return metrics.

    Regime strategies also overlay B0 (SPXT) and B1 (EW3) benchmark curves.
    """
    from scripts.backtest.signals import load_walk_forward_signals
    from scripts.backtest.strategies import (
        BENCHMARK_STRATEGY_KEYS,
        DATA_DRIVEN_14_STRATEGY_K,
        DATA_DRIVEN_STRATEGY_K,
        STRATEGY_BUILDERS,
    )

    if returns_panel is None:
        returns_panel = load_backtest_panel()
    if signals_k3 is None:
        signals_k3 = load_walk_forward_signals(3)
    if signals_k4 is None:
        signals_k4 = load_walk_forward_signals(4)
    if signals_k5 is None:
        signals_k5 = load_walk_forward_signals(5)

    builder = STRATEGY_BUILDERS[strategy_key]
    results: dict[str, pd.DataFrame] = {}
    metrics_rows: list[pd.Series] = []

    def _backtest_panel_for(strategy_key: str) -> pd.DataFrame:
        if strategy_key == "buy_and_hold_ew14" or strategy_key in DATA_DRIVEN_14_STRATEGY_K:
            return load_ew14_backtest_panel()
        return returns_panel

    def _run(sig: Any, tag: str, *, panel: pd.DataFrame | None = None) -> None:
        bt_panel = panel if panel is not None else _backtest_panel_for(strategy_key)
        if strategy_key in BENCHMARK_STRATEGY_KEYS:
            w = builder(sig, soft=False)
        else:
            w = builder(sig, soft=bool(soft))
        bt = run_strategy_backtest(w, bt_panel)
        results[tag] = bt
        metrics_rows.append(compute_metrics(bt, monthly_contribution=monthly_contribution, label=tag))

    if strategy_key in BENCHMARK_STRATEGY_KEYS:
        _run(signals_k4, "K=3 / K=4 / K=5 (same)")
    elif strategy_key in DATA_DRIVEN_STRATEGY_K:
        k_dd = DATA_DRIVEN_STRATEGY_K[strategy_key]
        sig_map = {3: signals_k3, 4: signals_k4, 5: signals_k5}
        _append_benchmark_curves(
            results,
            metrics_rows,
            returns_panel=returns_panel,
            signal=sig_map[k_dd],
            monthly_contribution=monthly_contribution,
        )
        _run(sig_map[k_dd], f"K={k_dd} {'soft' if soft else 'hard'}")
    elif strategy_key in DATA_DRIVEN_14_STRATEGY_K:
        from scripts.backtest.strategies import EW14_BENCHMARK_CURVE_LABELS

        k_dd = DATA_DRIVEN_14_STRATEGY_K[strategy_key]
        sig_map = {3: signals_k3, 4: signals_k4, 5: signals_k5}
        sig = sig_map[k_dd]
        ew14_panel = load_ew14_backtest_panel()
        common = ew14_panel.index.intersection(sig.index)
        ew14_panel = ew14_panel.reindex(common)

        for bench_key, bench_label in EW14_BENCHMARK_CURVE_LABELS.items():
            if bench_label in results:
                continue
            w_bench = STRATEGY_BUILDERS[bench_key](sig, soft=False)
            bt_bench = run_strategy_backtest(w_bench, ew14_panel)
            results[bench_label] = bt_bench
            metrics_rows.append(
                compute_metrics(bt_bench, monthly_contribution=monthly_contribution, label=bench_label)
            )

        w = builder(sig, soft=bool(soft))
        bt = run_strategy_backtest(w, ew14_panel)
        tag = f"K={k_dd} {'soft' if soft else 'hard'}"
        results[tag] = bt
        metrics_rows.append(
            compute_metrics(bt, monthly_contribution=monthly_contribution, label=tag)
        )
    else:
        _append_benchmark_curves(
            results,
            metrics_rows,
            returns_panel=returns_panel,
            signal=signals_k4,
            monthly_contribution=monthly_contribution,
        )
        if include_k3:
            _run(signals_k3, f"K=3 {'soft' if soft else 'hard'}")
        _run(signals_k4, f"K=4 {'soft' if soft else 'hard'}")
        _run(signals_k5, f"K=5 {'soft' if soft else 'hard'}")

    metrics_df = pd.DataFrame(metrics_rows).set_index("Label")
    mode = "soft" if soft else "hard" if soft is not None else ""
    plot_title = title or f"{strategy_key.replace('_', ' ').title()} ({mode})".strip()
    plot_strategy_report(results, metrics_df, title=plot_title, monthly_contribution=monthly_contribution)
    return metrics_df


def run_data_driven_overview_cell(
    *,
    returns_panel: pd.DataFrame | None = None,
    signals_k3: Any = None,
    signals_k4: Any = None,
    signals_k5: Any = None,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
    title: str | None = None,
    soft: bool = False,
) -> pd.DataFrame:
    """
    Single comparison plot: B0, B1, and data_driven_3/4/5 (each on matching K signals).

    Set ``soft=True`` for probability-weighted portfolio blends.
    """
    from scripts.backtest.signals import load_walk_forward_signals
    from scripts.backtest.strategies import (
        BENCHMARK_CURVE_LABELS,
        DATA_DRIVEN_STRATEGY_K,
        STRATEGY_BUILDERS,
    )

    mode = "soft" if soft else "hard"
    mode_label = "probability-weighted" if soft else "hard"
    if title is None:
        title = f"Data-driven ({mode_label}) — K=3 / K=4 / K=5 vs benchmarks"

    if returns_panel is None:
        returns_panel = load_backtest_panel()
    if signals_k3 is None:
        signals_k3 = load_walk_forward_signals(3)
    if signals_k4 is None:
        signals_k4 = load_walk_forward_signals(4)
    if signals_k5 is None:
        signals_k5 = load_walk_forward_signals(5)

    sig_map = {3: signals_k3, 4: signals_k4, 5: signals_k5}
    results: dict[str, pd.DataFrame] = {}
    metrics_rows: list[pd.Series] = []

    for bench_key, bench_label in BENCHMARK_CURVE_LABELS.items():
        w = STRATEGY_BUILDERS[bench_key](signals_k4, soft=False)
        bt = run_strategy_backtest(w, returns_panel)
        results[bench_label] = bt
        m = compute_metrics(bt, monthly_contribution=monthly_contribution, label=bench_label)
        m["Strategy"] = bench_key
        m["Mode"] = "baseline"
        metrics_rows.append(m)

    for strategy_key, k in DATA_DRIVEN_STRATEGY_K.items():
        label = f"K={k} {mode_label}"
        w = STRATEGY_BUILDERS[strategy_key](sig_map[k], soft=soft)
        bt = run_strategy_backtest(w, returns_panel)
        results[label] = bt
        m = compute_metrics(bt, monthly_contribution=monthly_contribution, label=label)
        m["Strategy"] = strategy_key
        m["Mode"] = mode
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Label")
    plot_strategy_report(results, metrics_df, title=title, monthly_contribution=monthly_contribution)
    return metrics_df


def run_data_driven_14_overview_cell(
    *,
    returns_panel: pd.DataFrame | None = None,
    signals_k3: Any = None,
    signals_k4: Any = None,
    signals_k5: Any = None,
    monthly_contribution: float = DEFAULT_MONTHLY_CONTRIBUTION,
    title: str | None = None,
    soft: bool = False,
) -> pd.DataFrame:
    """
    Fourteen-asset data-driven overview: B2 (EW14) vs data_driven_3/4/5_14.

    Investable universe excludes VIX, USGG3M, LUACOAS (regime signals only).
    Set ``soft=True`` for probability-weighted portfolio blends.
    """
    from scripts.backtest.signals import load_walk_forward_signals
    from scripts.backtest.strategies import (
        DATA_DRIVEN_14_STRATEGY_K,
        EW14_BENCHMARK_CURVE_LABELS,
        STRATEGY_BUILDERS,
    )

    mode_label = "probability-weighted" if soft else "hard"
    if title is None:
        title = f"Data-driven 14 assets ({mode_label}) — K=3 / K=4 / K=5 vs EW14"

    if returns_panel is None:
        returns_panel = load_ew14_backtest_panel()
    if signals_k3 is None:
        signals_k3 = load_walk_forward_signals(3)
    if signals_k4 is None:
        signals_k4 = load_walk_forward_signals(4)
    if signals_k5 is None:
        signals_k5 = load_walk_forward_signals(5)

    sig_map = {3: signals_k3, 4: signals_k4, 5: signals_k5}
    results: dict[str, pd.DataFrame] = {}
    metrics_rows: list[pd.Series] = []

    bench_key, bench_label = next(iter(EW14_BENCHMARK_CURVE_LABELS.items()))
    w_bench = STRATEGY_BUILDERS[bench_key](signals_k4, soft=False)
    bt_bench = run_strategy_backtest(w_bench, returns_panel)
    results[bench_label] = bt_bench
    m_bench = compute_metrics(bt_bench, monthly_contribution=monthly_contribution, label=bench_label)
    m_bench["Strategy"] = bench_key
    m_bench["Mode"] = "baseline"
    metrics_rows.append(m_bench)

    for strategy_key, k in DATA_DRIVEN_14_STRATEGY_K.items():
        label = f"Data-driven 14 assets K={k} ({mode_label})"
        w = STRATEGY_BUILDERS[strategy_key](sig_map[k], soft=soft)
        bt = run_strategy_backtest(w, returns_panel)
        results[label] = bt
        m = compute_metrics(bt, monthly_contribution=monthly_contribution, label=label)
        m["Strategy"] = strategy_key
        m["Mode"] = "soft" if soft else "hard"
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Label")
    plot_strategy_report(results, metrics_df, title=title, monthly_contribution=monthly_contribution)
    return metrics_df
