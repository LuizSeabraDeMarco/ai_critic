"""
Performance Evaluator
=====================
Goes well beyond accuracy: computes a full suite of classification or
regression metrics, detects suspiciously-perfect CV scores, and flags
class-imbalance problems.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from ai_critic.core.base import BaseEvaluator
from ai_critic.core.types import DimensionResult, ProblemType, Verdict


class PerformanceEvaluator(BaseEvaluator):
    """
    Multi-metric performance evaluation.

    Classification → accuracy, F1-macro, precision-macro, recall-macro,
                     ROC-AUC (binary), Matthews Correlation Coefficient
    Regression     → R², RMSE, MAE, MAPE
    """

    name = "performance"
    weight = 1.5  # highest weight — it's the anchor

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        cv = self._make_cv(y)

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            return self._evaluate_classification(model, X, y, problem_type, cv)
        return self._evaluate_regression(model, X, y, cv)

    # ------------------------------------------------------------------
    def _evaluate_classification(self, model, X, y, problem_type, cv):
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "mcc": "matthews_corrcoef",
        }
        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            scoring["roc_auc"] = "roc_auc"

        results = cross_validate(clone(model), X, y, cv=cv, scoring=scoring)

        means = {k: float(np.mean(v)) for k, v in results.items() if k.startswith("test_")}
        stds  = {k.replace("test_", ""): float(np.std(v))
                 for k, v in results.items() if k.startswith("test_")}
        means = {k.replace("test_", ""): v for k, v in means.items()}

        # Primary composite score: weighted blend
        primary = (
            means.get("accuracy", 0) * 0.3
            + means.get("f1_macro", 0) * 0.4
            + means.get("mcc", 0) * 0.3
        )
        primary = max(0.0, min(1.0, primary))

        suspicious = means.get("accuracy", 0) > 0.997

        suggestions = []
        if suspicious:
            suggestions.append("CV accuracy is suspiciously perfect — check for data leakage.")
        if means.get("f1_macro", 1) < 0.60:
            suggestions.append("F1-macro is low. Consider resampling, class weights, or a different algorithm.")
        if stds.get("accuracy", 0) > 0.05:
            suggestions.append("High CV variance in accuracy — the model may be unstable.")

        return DimensionResult(
            name=self.name,
            score=primary,
            verdict=Verdict.WARNING if suspicious else self._score_to_verdict(primary),
            summary=(
                f"Accuracy {means.get('accuracy', 0):.3f} | "
                f"F1-macro {means.get('f1_macro', 0):.3f} | "
                f"MCC {means.get('mcc', 0):.3f}"
                + (" ⚠ SUSPICIOUS" if suspicious else "")
            ),
            details={
                "means": means,
                "stds": stds,
                "cv_folds": cv.get_n_splits(),
                "suspiciously_perfect": suspicious,
            },
            suggestions=suggestions,
        )

    def _evaluate_regression(self, model, X, y, cv):
        from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

        def rmse(y_true, y_pred):
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))

        def mape(y_true, y_pred):
            mask = y_true != 0
            if not mask.any():
                return 0.0
            return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

        scoring = {
            "r2": "r2",
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            "rmse": make_scorer(rmse, greater_is_better=False),
            "mape": make_scorer(mape, greater_is_better=False),
        }

        results = cross_validate(clone(model), X, y, cv=cv, scoring=scoring)
        means = {k.replace("test_", ""): float(np.mean(np.abs(v)))
                 for k, v in results.items() if k.startswith("test_")}

        r2 = float(np.mean(results["test_r2"]))
        primary = max(0.0, min(1.0, r2))

        suggestions = []
        if r2 < 0.5:
            suggestions.append("R² < 0.5 — the model explains less than half the variance. Try feature engineering.")
        if means.get("mape", 0) > 0.30:
            suggestions.append("MAPE > 30% — consider log-transforming the target.")

        return DimensionResult(
            name=self.name,
            score=primary,
            verdict=self._score_to_verdict(primary, warn_below=0.70, fail_below=0.40),
            summary=f"R² {r2:.3f} | MAE {means.get('mae', 0):.4f} | RMSE {means.get('rmse', 0):.4f}",
            details={"means": means, "cv_folds": cv.get_n_splits()},
            suggestions=suggestions,
        )
