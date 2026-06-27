"""
Explainability Evaluator
========================
Uses permutation importance to measure how much the model depends
on each feature. Flags over-reliance on a single feature (leakage proxy)
and provides feature names when available.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ai_critic.core.base import BaseEvaluator
from ai_critic.core.types import DimensionResult, ProblemType, Verdict


class ExplainabilityEvaluator(BaseEvaluator):

    name = "explainability"
    weight = 0.9
    depends_on = ["performance"]

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, Any]] = None,
    ) -> DimensionResult:

        feature_names: Optional[List[str]] = (context or {}).get("feature_names")

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
            m = clone(model).fit(X_tr, y_tr)
            result = permutation_importance(
                m, X_val, y_val, n_repeats=5, random_state=42
            )
        except Exception as exc:
            return DimensionResult(
                name=self.name,
                score=0.5,
                verdict=Verdict.WARNING,
                summary=f"Permutation importance failed: {exc}",
                details={"error": str(exc)},
                suggestions=["Ensure model is sklearn-compatible and supports .predict()."],
            )

        importances = result.importances_mean
        total = importances.sum()
        if total <= 0:
            normalized = np.ones(len(importances)) / len(importances)
        else:
            normalized = importances / total

        # Gini concentration index (1 = dominated by one feature, 0 = perfectly equal)
        sorted_imp = np.sort(normalized)
        n = len(sorted_imp)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_imp))) / (n * sorted_imp.sum() + 1e-9) - (n + 1) / n
        gini = float(np.clip(gini, 0, 1))

        # Score: penalise high concentration
        score = max(0.0, 1.0 - gini)

        # Top features with names
        top_n = min(10, len(normalized))
        top_idx = np.argsort(normalized)[::-1][:top_n]
        if feature_names:
            top_features = {
                feature_names[i]: round(float(normalized[i]), 4)
                for i in top_idx if i < len(feature_names)
            }
        else:
            top_features = {
                f"feature_{i}": round(float(normalized[i]), 4)
                for i in top_idx
            }

        dominant = max(normalized) if len(normalized) > 0 else 0.0
        dominant_name = (
            feature_names[int(np.argmax(normalized))]
            if feature_names else f"feature_{int(np.argmax(normalized))}"
        )

        suggestions = []
        if gini > 0.70:
            suggestions.append(
                f"High feature concentration (Gini {gini:.2f}) — '{dominant_name}' "
                "dominates. Check for data leakage or consider feature engineering."
            )
        if dominant > 0.90:
            suggestions.append(
                f"'{dominant_name}' accounts for {dominant:.0%} of importance — "
                "the model may be effectively a single-feature rule."
            )

        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score, warn_below=0.60, fail_below=0.35),
            summary=f"Gini concentration {gini:.2f} | Top feature: '{dominant_name}' ({dominant:.0%})",
            details={
                "gini_concentration": round(gini, 4),
                "top_features": top_features,
                "n_features": X.shape[1],
            },
            suggestions=suggestions,
        )