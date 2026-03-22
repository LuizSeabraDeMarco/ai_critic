import numpy as np
from collections import Counter
from typing import Dict, Any

from ai_critic.critic import AICritic


def _detect_class_imbalance(y) -> Dict[str, Any]:
    counts = Counter(y)
    if len(counts) <= 1:
        return {"risk": "HIGH", "message": "Dataset contains only one class."}

    max_count = max(counts.values())
    min_count = min(counts.values())
    ratio = max_count / min_count

    if ratio > 10:
        risk, msg = "HIGH", "Severe class imbalance detected."
    elif ratio > 3:
        risk, msg = "MEDIUM", "Moderate class imbalance detected."
    else:
        risk, msg = "LOW", "Class distribution looks balanced."

    return {"risk": risk, "ratio": float(ratio), "message": msg}


def _detect_dataset_size(X) -> Dict[str, Any]:
    n_samples = len(X)
    if n_samples < 100:
        risk, msg = "HIGH", "Very small dataset."
    elif n_samples < 1000:
        risk, msg = "MEDIUM", "Dataset may be small for complex models."
    else:
        risk, msg = "LOW", "Dataset size looks reasonable."

    return {"samples": n_samples, "risk": risk, "message": msg}


def _detect_overfitting(performance_result: Dict[str, Any]) -> Dict[str, Any]:
    std = performance_result.get("cv_std", 0)
    mean = performance_result.get("cv_mean_score", 0)

    if mean > 0.98 and std < 0.01:
        risk, msg = "HIGH", "Model may be memorizing the dataset."
    elif std > 0.1:
        risk, msg = "MEDIUM", "Model performance unstable across folds."
    else:
        risk, msg = "LOW", "No strong overfitting signals detected."

    return {"risk": risk, "message": msg}


def _detect_possible_leakage(performance_result: Dict[str, Any]) -> Dict[str, Any]:
    if performance_result.get("suspiciously_perfect"):
        return {"risk": "HIGH", "message": "Suspiciously perfect cross-validation score."}
    return {"risk": "LOW", "message": "No obvious leakage indicators."}


def audit(model, X, y) -> Dict[str, Any]:
    """
    Run a full AI audit combining evaluation pipeline and dataset checks.
    """
    critic = AICritic()
    report = critic.evaluate(model, X, y)

    performance = report.details.get("performance", {})

    dataset_checks = {
        "size": _detect_dataset_size(X),
        "class_imbalance": _detect_class_imbalance(y)
    }

    model_checks = {
        "overfitting": _detect_overfitting(performance),
        "data_leakage": _detect_possible_leakage(performance)
    }

    # Combine all into a comprehensive audit result
    return {
        "report": report.to_dict(),
        "audit_checks": {
            "dataset": dataset_checks,
            "model": model_checks
        },
        "summary": report.summary()
    }
