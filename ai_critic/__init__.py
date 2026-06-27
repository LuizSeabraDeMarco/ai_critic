"""
ai_critic
===========
Comprehensive ML model evaluation that goes far beyond accuracy.

Quick start
-----------
>>> import ai_critic
>>> report = ai_critic.audit(model, X, y)
>>> print(report.overall_score, report.overall_verdict)

Works with numpy arrays *and* pandas DataFrames:
>>> import pandas as pd
>>> report = ai_critic.audit(model, df.drop("target", axis=1), df["target"])
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ai_critic.core.pipeline import AuditPipeline
from ai_critic.core.types import AuditReport
from ai_critic.evaluators.performance import PerformanceEvaluator
from ai_critic.evaluators.robustness import RobustnessEvaluator
from ai_critic.evaluators.explainability import ExplainabilityEvaluator
from ai_critic.evaluators.data_quality import DataQualityEvaluator
from ai_critic.evaluators.fairness import FairnessEvaluator
from ai_critic.evaluators.calibration import CalibrationEvaluator
from ai_critic.evaluators.complexity import ComplexityEvaluator

__version__ = "2.0.0"
__all__ = ["audit", "gate", "AuditPipeline", "AuditReport", "__version__"]

_DEFAULT_EVALUATORS = [
    DataQualityEvaluator(),
    ComplexityEvaluator(),
    PerformanceEvaluator(),
    RobustnessEvaluator(),
    ExplainabilityEvaluator(),
    CalibrationEvaluator(),
    FairnessEvaluator(),
]


def audit(
    model: Any,
    X: Any,
    y: Any,
    *,
    weights: Optional[Dict[str, float]] = None,
    parallel: bool = False,
    evaluators: Optional[List] = None,
    sensitive_features: Optional[List[str]] = None,
) -> AuditReport:
    """
    Run a full model audit and return an AuditReport.

    Parameters
    ----------
    model              : fitted sklearn-compatible estimator (also works with
                         LightGBM, XGBoost, CatBoost sklearn wrappers)
    X                  : feature matrix — numpy array *or* pandas DataFrame
    y                  : target vector — numpy array or pandas Series
    weights            : optional dict to override per-dimension weight,
                         e.g. {"fairness": 2.0, "robustness": 1.5}
    parallel           : run independent evaluators concurrently (ThreadPool)
    evaluators         : custom list of BaseEvaluator instances
                         (overrides the 7 default dimensions)
    sensitive_features : list of column names (DataFrame) or indices (array)
                         to force into the fairness evaluator, e.g. ["gender", "age_group"]

    Returns
    -------
    AuditReport
        .overall_score    float 0–1
        .overall_verdict  Verdict enum  (PASS / WARNING / FAIL)
        .dimensions       dict[str, DimensionResult]
        .top_suggestions  list[str]
    """
    # ── Normalise inputs ────────────────────────────────────────────
    feature_names: Optional[List[str]] = None

    try:
        import pandas as pd  # soft dependency — not in install_requires

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
    except ImportError:
        pass

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # ── Resolve sensitive columns ───────────────────────────────────
    sensitive_indices: Optional[List[int]] = None
    if sensitive_features is not None and feature_names is not None:
        sensitive_indices = [
            feature_names.index(f) for f in sensitive_features if f in feature_names
        ]
    elif sensitive_features is not None:
        # Passed as integer indices already
        sensitive_indices = [int(i) for i in sensitive_features]

    # Inject sensitive_indices into FairnessEvaluator if present
    evs = evaluators or _DEFAULT_EVALUATORS
    if sensitive_indices is not None:
        evs = [
            ev.with_sensitive_columns(sensitive_indices)
            if hasattr(ev, "with_sensitive_columns") else ev
            for ev in evs
        ]

    pipeline = AuditPipeline(
        evaluators=evs,
        weights=weights,
        feature_names=feature_names,
    )
    return pipeline.run(model, X, y, parallel=parallel)


def gate(report: AuditReport, min_score: float = 0.70) -> None:
    """
    Raise RuntimeError if overall_score < min_score.

    Use in CI/CD pipelines to block models that don't meet the bar:

    >>> report = ai_critic.audit(model, X, y)
    >>> ai_critic.gate(report, min_score=0.75)   # raises if fails
    """
    if report.overall_score < min_score:
        issues = "\n".join(f"  - {s}" for s in report.top_suggestions[:5])
        raise RuntimeError(
            f"Model audit FAILED: overall score {report.overall_score:.3f} "
            f"is below threshold {min_score}.\n"
            f"Top issues:\n{issues}"
        )