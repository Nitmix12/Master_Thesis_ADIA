from sklearn.mixture import BayesianGaussianMixture
import numpy as np

def get_active_regimes(data, max_components=10, weight_threshold=0.05, covariance_type='full', random_state=None, alpha_prior=None):
    """
    Fits a Bayesian GMM (Dirichlet Process) and returns the number of active regimes.
    weight_threshold: Regimes with weights below this (e.g., 5%) are ignored.
    """
    bgmm = BayesianGaussianMixture(
        n_components=max_components, 
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=alpha_prior,
        covariance_type=covariance_type, 
        max_iter=2000,             # Increased to 2000 to ensure convergence
        random_state=random_state  # Allows fixed state for plots, random for bootstrap
    )
    bgmm.fit(data)
    
    # Count how many weights are above the threshold
    active_count = sum(bgmm.weights_ > weight_threshold)
    return active_count, bgmm.weights_