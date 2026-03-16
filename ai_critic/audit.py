import numpy as np
from collections import Counter

from ai_critic.critic import AICritic


def _detect_class_imbalance(y):

    counts = Counter(y)

    if len(counts) <= 1:
        return {
            "risk": "HIGH",
            "message": "Dataset contains only one class."
        }

    max_count = max(counts.values())
    min_count = min(counts.values())

    ratio = max_count / min_count

    if ratio > 10:
        risk = "HIGH"
        msg = "Severe class imbalance detected."
    elif ratio > 3:
        risk = "MEDIUM"
        msg = "Moderate class imbalance detected."
    else:
        risk = "LOW"
        msg = "Class distribution looks balanced."

    return {
        "risk": risk,
        "ratio": ratio,
        "message": msg
    }


def _detect_dataset_size(X):

    n_samples = len(X)

    if n_samples < 100:
        risk = "HIGH"
        msg = "Very small dataset."
    elif n_samples < 1000:
        risk = "MEDIUM"
        msg = "Dataset may be small for complex models."
    else:
        risk = "LOW"
        msg = "Dataset size looks reasonable."

    return {
        "samples": n_samples,
        "risk": risk,
        "message": msg
    }


def _detect_overfitting(performance_result):

    std = performance_result.get("cv_std", 0)
    mean = performance_result.get("cv_mean_score", 0)

    if mean > 0.98 and std < 0.01:
        risk = "HIGH"
        msg = "Model may be memorizing the dataset."
    elif std > 0.1:
        risk = "MEDIUM"
        msg = "Model performance unstable across folds."
    else:
        risk = "LOW"
        msg = "No strong overfitting signals detected."

    return {
        "risk": risk,
        "message": msg
    }


def _detect_possible_leakage(performance_result):

    if performance_result.get("suspiciously_perfect"):
        return {
            "risk": "HIGH",
            "message": "Suspiciously perfect cross-validation score."
        }

    return {
        "risk": "LOW",
        "message": "No obvious leakage indicators."
    }


def audit(model, X, y):
    """
    Run a full AI audit combining evaluation pipeline and dataset checks.
    """

    critic = AICritic()

    evaluation = critic.evaluate(model, X, y)

    performance = evaluation["details"].get("performance", {})

    dataset_checks = {
        "size": _detect_dataset_size(X),
        "class_imbalance": _detect_class_imbalance(y)
    }

    model_checks = {
        "overfitting": _detect_overfitting(performance),
        "data_leakage": _detect_possible_leakage(performance)
    }

    return {
        "scores": evaluation["scores"],
        "evaluation_details": evaluation["details"],
        "dataset_checks": dataset_checks,
        "model_checks": model_checks
    }