"""
model_audit
===========
Comprehensive ML model evaluation that goes far beyond accuracy.

Quick start
-----------
>>> import model_audit
>>> report = model_audit.audit(model, X, y)
>>> print(report.overall_score, report.overall_verdict)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from model_audit.core.pipeline import AuditPipeline
from model_audit.core.types import AuditReport
from model_audit.evaluators.performance import PerformanceEvaluator
from model_audit.evaluators.robustness import RobustnessEvaluator
from model_audit.evaluators.explainability import ExplainabilityEvaluator
from model_audit.evaluators.data_quality import DataQualityEvaluator
from model_audit.evaluators.fairness import FairnessEvaluator
from model_audit.evaluators.calibration import CalibrationEvaluator
from model_audit.evaluators.complexity import ComplexityEvaluator

__version__ = "1.0.0"
__all__ = ["audit", "AuditPipeline", "AuditReport"]

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
    X: np.ndarray,
    y: np.ndarray,
    *,
    weights: Optional[Dict[str, float]] = None,
    parallel: bool = False,
    evaluators: Optional[List] = None,
) -> AuditReport:
    """
    Run a full model audit and return an AuditReport.

    Parameters
    ----------
    model   : fitted sklearn-compatible estimator
    X       : feature matrix (n_samples × n_features)
    y       : target vector (n_samples,)
    weights : optional dict to override per-dimension weight
    parallel: run independent evaluators concurrently
    evaluators: custom list of BaseEvaluator instances (overrides defaults)

    Returns
    -------
    AuditReport with .overall_score, .overall_verdict, .dimensions, .top_suggestions
    """
    pipeline = AuditPipeline(
        evaluators=evaluators or _DEFAULT_EVALUATORS,
        weights=weights,
    )
    return pipeline.run(model, X, y, parallel=parallel)


def gate(report: AuditReport, min_score: float = 0.70) -> None:
    """
    Raise RuntimeError if overall_score < min_score.
    Use in CI/CD pipelines to block bad models.
    """
    if report.overall_score < min_score:
        raise RuntimeError(
            f"Model audit FAILED: overall score {report.overall_score:.3f} "
            f"is below threshold {min_score}.\n"
            f"Top issues:\n" + "\n".join(f"  - {s}" for s in report.top_suggestions[:5])
        )
