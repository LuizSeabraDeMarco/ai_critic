import math

def evaluate(model, n_samples=None, n_features=None):
    params = model.get_params()
    model_type = type(model).__name__

    report = {
        "model_type": model_type,
        "n_params": len(params),
        "uses_random_state": "random_state" in params
    }

    # 🧠 Structural overfitting heuristics
    warnings = []

    if n_samples and hasattr(model, "max_depth"):
        max_depth = params.get("max_depth")
        if max_depth is not None:
            recommended_depth = math.log2(n_samples)
            if max_depth > recommended_depth:
                warnings.append({
                    "issue": "structural_overfitting_risk",
                    "max_depth": max_depth,
                    "recommended_max_depth": int(recommended_depth),
                    "message": "Tree depth may be too high for dataset size."
                })

    if n_samples and n_features and n_features > n_samples:
        warnings.append({
            "issue": "high_feature_sample_ratio",
            "message": "More features than samples can cause instability."
        })

    report["structural_warnings"] = warnings
    return report
