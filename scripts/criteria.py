import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def calculate_icl(bic_value, proba_matrix):
    """
    Calculates the Integrated Completed Likelihood (ICL).
    ICL = BIC + 2 * Entropy.
    """
    eps = 1e-15
    proba_matrix = np.clip(proba_matrix, eps, 1 - eps)
    entropy = -np.sum(proba_matrix * np.log(proba_matrix))
    return bic_value + (2 * entropy)

def run_information_criteria(model_factory, X, k_range=range(2, 10), model_name="Model"):
    """
    Evaluates AIC, BIC, and ICL for a range of K.
    
    Parameters:
    - model_factory: A function that takes 'k' and returns a FITTED model.
                     The returned model must have .aic(X), .bic(X), and .predict_proba(X)
    - X: The scaled feature data.
    - k_range: Iterable of regime counts to test.
    - model_name: String for the plot title.
    """
    aic_scores = []
    bic_scores = []
    icl_scores = []

    print(f"Running optimization for {model_name} with K={list(k_range)}...")

    for k in k_range:
        # 1. Build and fit the model using your custom logic
        model = model_factory(k)
        
        # 2. Extract criteria
        aic = model.aic(X)
        bic = model.bic(X)
        probs = model.predict_proba(X)
        icl = calculate_icl(bic, probs)
        
        aic_scores.append(aic)
        bic_scores.append(bic)
        icl_scores.append(icl)

    # 3. Create a DataFrame
    df_results = pd.DataFrame({
        'K': list(k_range),
        'AIC': aic_scores,
        'BIC': bic_scores,
        'ICL': icl_scores
    }).set_index('K')

    # Display the table cleanly in the notebook
    display(df_results)

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Number of Regimes (K)")
    ax1.set_ylabel("Scores (BIC / ICL)", color="black")

    line1, = ax1.plot(k_range, bic_scores, marker="o", color="tab:blue", label="BIC")
    line2, = ax1.plot(k_range, icl_scores, marker="^", color="tab:green", label="ICL")

    ax2 = ax1.twinx()
    ax2.set_ylabel("AIC Score", color="tab:red")
    line3, = ax2.plot(k_range, aic_scores, marker="s", color="tab:red", label="AIC")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"Regime Optimization ({model_name}): BIC, AIC, and ICL")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Renders inline in the Jupyter Notebook
    plt.show()

    # 5. Output Optimal Ks
    print(f"Optimal K (AIC): {k_range[np.argmin(aic_scores)]}")
    print(f"Optimal K (BIC): {k_range[np.argmin(bic_scores)]}")
    print(f"Optimal K (ICL): {k_range[np.argmin(icl_scores)]}")
    
    return df_results