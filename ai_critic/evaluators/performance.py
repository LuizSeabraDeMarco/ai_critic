from sklearn.model_selection import cross_val_score

def evaluate(model, X, y):
    scores = cross_val_score(model, X, y, cv=3)

    mean = float(scores.mean())
    std = float(scores.std())

    suspicious = mean > 0.995

    return {
        "cv_mean_score": mean,
        "cv_std": std,
        "suspiciously_perfect": suspicious,
        "message": (
            "Perfect CV score detected — possible data leakage."
            if suspicious
            else "CV performance within expected range."
        )
    }
