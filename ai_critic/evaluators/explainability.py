# explainability.py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

from .validation import make_cv


def evaluate(model, X, y, max_features=10):
    """
    Model-agnostic feature sensitivity analysis.
    Measures how much performance drops when each feature is permuted.
    """

    cv = make_cv(y)

    base_model = clone(model)
    base_score = cross_val_score(base_model, X, y, cv=cv).mean()

    sensitivities = []

    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])

        permuted_model = clone(model)
        score = cross_val_score(permuted_model, X_permuted, y, cv=cv).mean()

        drop = base_score - score

        sensitivities.append({
            "feature_index": int(i),
            "performance_drop": float(drop)
        })

    sensitivities.sort(
        key=lambda x: x["performance_drop"],
        reverse=True
    )

    top = sensitivities[:max_features]

    verdict = "stable"
    message = "No single feature dominates model behavior."

    if top and top[0]["performance_drop"] > 0.30:
        verdict = "feature_leakage_risk"
        message = (
            "Model is highly sensitive to a single feature, "
            "which may indicate leakage or shortcut learning."
        )
    elif top and top[0]["performance_drop"] > 0.15:
        verdict = "feature_dependency"
        message = (
            "Model depends strongly on a small subset of features."
        )

    return {
        "baseline_score": float(base_score),
        "top_sensitive_features": top,
        "max_performance_drop": float(top[0]["performance_drop"]) if top else 0.0,
        "verdict": verdict,
        "message": message
    }
