# evaluators/config.py
import math


def evaluate(model, n_samples=None, n_features=None):
    """
    Evaluates model configuration for structural risks and complexity.
    Outputs only metadata-safe signals (telemetry-ready).
    """

    params = model.get_params() if hasattr(model, "get_params") else {}
    model_type = type(model).__name__

    report = {
        "model_type": model_type,
        "n_params": len(params),
        "uses_random_state": "random_state" in params,
        "complexity_score": 0,
        "risk_level": "low",
    }

    warnings = []

    # =========================
    # Tree depth heuristic
    # =========================
    if n_samples and "max_depth" in params:
        max_depth = params.get("max_depth")
        if max_depth is not None:
            recommended_depth = math.log2(max(2, n_samples))
            if max_depth > recommended_depth:
                warnings.append({
                    "issue": "structural_overfitting_risk",
                    "max_depth": max_depth,
                    "recommended_max_depth": int(recommended_depth),
                    "message": "Tree depth may be too high for dataset size."
                })
                report["complexity_score"] += 1

    # =========================
    # Feature / sample ratio
    # =========================
    if n_samples and n_features and n_features > n_samples:
        warnings.append({
            "issue": "high_feature_sample_ratio",
            "message": "More features than samples can cause instability."
        })
        report["complexity_score"] += 1

    # =========================
    # Risk aggregation
    # =========================
    if report["complexity_score"] >= 2:
        report["risk_level"] = "high"
    elif report["complexity_score"] == 1:
        report["risk_level"] = "medium"

    report["structural_warnings"] = warnings
    return report
