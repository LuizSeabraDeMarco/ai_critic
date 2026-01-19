import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def evaluate(model, X, y, leakage_suspected=False):
    noise_level = 0.02  # 2% relative noise

    scale = np.std(X)
    noise = np.random.normal(0, noise_level * scale, X.shape)
    X_noisy = X + noise

    model_clean = clone(model)
    model_noisy = clone(model)

    score_clean = cross_val_score(
        model_clean, X, y, cv=3, n_jobs=1
    ).mean()

    score_noisy = cross_val_score(
        model_noisy, X_noisy, y, cv=3, n_jobs=1
    ).mean()

    drop = score_clean - score_noisy

    # =========================
    # Semantic robustness logic
    # =========================
    if leakage_suspected and score_clean > 0.98:
        verdict = "misleading"
        message = (
            "Model appears robust to noise, but original performance is "
            "likely inflated due to data leakage."
        )
    elif drop > 0.15:
        verdict = "fragile"
        message = "Model performance degrades significantly under noise."
    else:
        verdict = "stable"
        message = "Model shows acceptable robustness to noise."

    return {
        "cv_score_original": float(score_clean),
        "cv_score_noisy": float(score_noisy),
        "performance_drop": float(drop),
        "verdict": verdict,
        "message": message
    }
