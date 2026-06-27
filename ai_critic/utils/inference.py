"""Utilities for automatic inference of dataset properties."""
from __future__ import annotations

import numpy as np

from ai_critic.core.types import ProblemType


def infer_problem_type(y: np.ndarray) -> ProblemType:
    y = np.asarray(y)
    unique = np.unique(y)
    n_unique = len(unique)

    if np.issubdtype(y.dtype, np.floating) and n_unique > 20:
        return ProblemType.REGRESSION
    if n_unique == 2:
        return ProblemType.BINARY_CLASSIFICATION
    if n_unique <= 50:
        return ProblemType.MULTICLASS_CLASSIFICATION
    return ProblemType.REGRESSION


def is_classifier(model) -> bool:
    from sklearn.base import is_classifier as sk_is_classifier
    try:
        return sk_is_classifier(model)
    except Exception:
        return hasattr(model, "predict_proba")
