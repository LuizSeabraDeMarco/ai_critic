import numpy as np

def evaluate(X, y):
    report = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "has_nan": bool(np.isnan(X).any() or np.isnan(y).any())
    }

    # Class balance (JSON-safe)
    if len(set(y)) < 20:
        values, counts = np.unique(y, return_counts=True)
        report["class_balance"] = {
            int(v): int(c) for v, c in zip(values, counts)
        }
    else:
        report["class_balance"] = "many_classes"

    # 🔎 Data leakage detection (high correlation with target)
    suspicious_features = []

    y_mean = np.mean(y)
    y_centered = y - y_mean

    for i in range(X.shape[1]):
        feature = X[:, i]

        if np.std(feature) == 0:
            continue

        corr = np.corrcoef(feature, y_centered)[0, 1]

        if abs(corr) > 0.98:
            suspicious_features.append({
                "feature_index": int(i),
                "correlation": float(corr)
            })

    report["data_leakage"] = {
        "suspected": bool(len(suspicious_features) > 0),
        "details": suspicious_features,
        "message": (
            "Highly correlated features may reveal the target directly."
            if suspicious_features
            else "No obvious data leakage detected."
        )
    }

    return report
