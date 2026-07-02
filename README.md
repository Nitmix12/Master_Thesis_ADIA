# pmr_paper — Botte & Bao GMM reproduction (ADIA)

Self-contained pipeline for **4- and 5-regime Gaussian mixture models** on 17 Bloomberg macro/style factors.

## Workflow

```bash
# 1. Build monthly factor returns
python scripts/data_preparation.py
# → data/features.csv

# 2. Model selection (optional)
# notebooks/model_selection/

# 3. Regime models + paper figures
# notebooks/models/01_static_gmm.ipynb
# notebooks/models/02_walk_forward_gmm.ipynb
```

## GMM fitting

- **K=4:** `covariance_type="full"`, `n_init=20` (static and walk-forward)
- **K=5:** `covariance_type="diag"`, `n_init=20` (static and walk-forward; matches old/05_GMM_5reg)
- **K=5 static labels:** optional trailing 3-month mode smoothing (`center=False`; no future-month look-ahead in the rolling window)
- **K=5 walk-forward:** dwell hysteresis + causal EMA only; island suppression is **off** by default (pass `use_island_suppression=True` in `run_walk_forward` for optional plot smoothing)

## Regime labeling

1. **Economic (template) Hungarian** — match GMM components to {Crisis, Inflation, Steady State, Walking on Ice, [Bull]} using sign templates on cluster means/volatilities (old Bloomberg v3 / 5-reg, **not** v1 softened templates).
2. **Temporal Hungarian** (walk-forward only) — keep component IDs stable across monthly refits in **raw** feature space.

## Fifth regime: Bull Market

**Bull Market** is kept (not replaced) because it separates **low-volatility equity rallies** (e.g. late 1990s) from **Steady State** without overlapping **Walking on Ice** (elevated vol, fragile risk-on). Alternatives like “Recovery” or “Expansion” are harder to distinguish from Steady State in the 17-factor space.

## Outputs

- `data/outputs/` — regime CSVs  
- `outputs/figures/` — static and walk-forward plots  

## Strategy backtests

```bash
# notebooks/backtest/01_strategy_comparison.ipynb
```

Uses `walk_forward_k4.csv` / `walk_forward_k5.csv` with hard + soft regime rules (buy & hold, risk-on/off, safe haven, all-weather, inverse vol).
