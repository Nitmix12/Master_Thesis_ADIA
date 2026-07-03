from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scripts.bayesian import get_active_regimes
from scripts.paths import OUTPUT_DIR

DEFAULT_BOOTSTRAP_SEED = 42


def get_random_block(data, block_size, rng):
    """Grabs a single continuous block of time-series data."""
    n_samples = data.shape[0]
    max_start = n_samples - block_size
    start_idx = rng.integers(0, max_start + 1)
    return data[start_idx : start_idx + block_size]


def get_stitched_block_bootstrap(data, block_size, rng):
    """
    Constructs a full-length dataset by stitching random blocks together.
    This preserves the total sample size (N) but breaks the specific historical timeline.
    """
    n_samples = data.shape[0]
    n_blocks = int(np.ceil(n_samples / block_size))
    
    blocks = []
    for _ in range(n_blocks):
        blocks.append(get_random_block(data, block_size, rng))
    
    # Concatenate and truncate exactly to the original length
    bootstrapped_data = np.vstack(blocks)[:n_samples]
    return bootstrapped_data


def _bootstrap_iteration(data, block_size, max_components, threshold, alpha, seed):
    """A single reproducible iteration of the bootstrap process."""
    rng = np.random.default_rng(int(seed))
    # 1. Generate the resampled data
    sample = get_stitched_block_bootstrap(data, block_size, rng)
    
    # 2. Fit the Bayesian GMM (full covariance — matches v1 model_selection bootstrap)
    active_k, _ = get_active_regimes(
        data=sample,
        max_components=max_components,
        weight_threshold=threshold,
        covariance_type="full",
        random_state=int(seed),
        alpha_prior=alpha
    )
    return int(active_k)


def _default_output_path(run_label: str | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_label}" if run_label else ""
    return OUTPUT_DIR / f"bayesian_bootstrap_raw_{timestamp}{suffix}.csv"


def run_bayesian_bootstrap(
    data,
    n_iterations=1000,
    block_size=120,
    max_components=10,
    threshold=0.04,
    alpha=0.1,
    n_jobs=-1,
    master_seed: int = DEFAULT_BOOTSTRAP_SEED,
    save_raw: bool = True,
    output_path: str | Path | None = None,
    run_label: str | None = None,
):
    """
    Runs the block bootstrap process in parallel across all CPU cores.

    Uses a master seed to generate one deterministic seed per iteration, which
    makes results reproducible across reruns and safe to parallelize. By
    default, raw iteration-level results are also saved to ``data/outputs/``.
    """
    print(
        f"Starting {n_iterations} bootstrap iterations "
        f"(Block Size: {block_size} months, master_seed={master_seed})..."
    )

    seed_sequence = np.random.SeedSequence(int(master_seed))
    iteration_seeds = seed_sequence.generate_state(n_iterations, dtype=np.uint32)
    
    # Run in parallel to save time
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_bootstrap_iteration)(
            data,
            block_size,
            max_components,
            threshold,
            alpha,
            int(seed),
        )
        for seed in iteration_seeds
    )

    if save_raw:
        path = Path(output_path) if output_path is not None else _default_output_path(run_label=run_label)
        raw = pd.DataFrame(
            {
                "iteration": np.arange(1, n_iterations + 1, dtype=int),
                "seed": iteration_seeds.astype(np.uint64),
                "active_regimes": np.asarray(results, dtype=int),
                "block_size": int(block_size),
                "max_components": int(max_components),
                "weight_threshold": float(threshold),
                "alpha_prior": float(alpha),
                "master_seed": int(master_seed),
            }
        )
        raw.to_csv(path, index=False)
        print(f"Saved raw bootstrap results to {path}")

    return results