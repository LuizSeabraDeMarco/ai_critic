"""
Abstract base class for all evaluators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ai_critic.core.types import DimensionResult, ProblemType


class BaseEvaluator(ABC):
    """
    Every evaluator must implement `evaluate()` and return a DimensionResult.

    Attributes
    ----------
    name : str
        Unique identifier used in the report.
    weight : float
        Contribution weight when computing the overall score (default 1.0).
    depends_on : list[str]
        Names of evaluators whose results this one needs (passed via ``context``).
    """

    name: str = "base"
    weight: float = 1.0
    depends_on: List[str] = []

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:
        """Run the evaluation and return a DimensionResult."""

    # ------------------------------------------------------------------
    # Helpers shared by multiple evaluators
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cv(y: np.ndarray, n_splits: int = 5):
        from sklearn.model_selection import KFold, StratifiedKFold
        from ai_critic.utils.inference import infer_problem_type, ProblemType as PT

        pt = infer_problem_type(y)
        if pt in (PT.BINARY_CLASSIFICATION, PT.MULTICLASS_CLASSIFICATION):
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    @staticmethod
    def _score_to_verdict(score: float, warn_below: float = 0.75, fail_below: float = 0.50):
        from ai_critic.core.types import Verdict
        if score >= warn_below:
            return Verdict.PASS
        if score >= fail_below:
            return Verdict.WARNING
        return Verdict.FAIL
