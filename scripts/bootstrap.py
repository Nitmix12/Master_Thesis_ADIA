import numpy as np
from joblib import Parallel, delayed
from scripts.bayesian import get_active_regimes

def get_random_block(data, block_size):
    """Grabs a single continuous block of time-series data."""
    n_samples = data.shape[0]
    max_start = n_samples - block_size
    start_idx = np.random.randint(0, max_start + 1)
    return data[start_idx : start_idx + block_size]

def get_stitched_block_bootstrap(data, block_size):
    """
    Constructs a full-length dataset by stitching random blocks together.
    This preserves the total sample size (N) but breaks the specific historical timeline.
    """
    n_samples = data.shape[0]
    n_blocks = int(np.ceil(n_samples / block_size))
    
    blocks = []
    for _ in range(n_blocks):
        blocks.append(get_random_block(data, block_size))
    
    # Concatenate and truncate exactly to the original length
    bootstrapped_data = np.vstack(blocks)[:n_samples]
    return bootstrapped_data

def _bootstrap_iteration(data, block_size, max_components, threshold, alpha):
    """A single iteration of the bootstrap process."""
    # 1. Generate the resampled data
    sample = get_stitched_block_bootstrap(data, block_size)
    
    # 2. Fit the Bayesian GMM (full covariance — matches v1 model_selection bootstrap)
    active_k, _ = get_active_regimes(
        data=sample,
        max_components=max_components,
        weight_threshold=threshold,
        covariance_type="full",
        random_state=None,       # MUST be None so it explores randomly
        alpha_prior=alpha
    )
    return active_k

def run_bayesian_bootstrap(data, n_iterations=1000, block_size=120, max_components=10, 
                           threshold=0.04, alpha=0.1, n_jobs=-1):
    """
    Runs the block bootstrap process in parallel across all CPU cores.
    """
    print(f"Starting {n_iterations} bootstrap iterations (Block Size: {block_size} months)...")
    
    # Run in parallel to save time
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_bootstrap_iteration)(data, block_size, max_components, threshold, alpha)
        for _ in range(n_iterations)
    )
    
    return results