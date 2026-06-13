"""
Explainability Evaluator
========================
Runs permutation importance over ALL features (not capped at 10),
detects concentration risk (model depends on very few features),
and reports a Gini-style concentration index.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from model_audit.core.base import BaseEvaluator
from model_audit.core.types import DimensionResult, ProblemType, Verdict


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
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        # Fit on full data for permutation importance
        fitted = clone(model).fit(X, y)

        pi = permutation_importance(
            fitted, X, y,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

        importances = pi.importances_mean
        importances_std = pi.importances_std

        # Normalise to [0, 1] for analysis
        total = importances.sum()
        if total <= 0:
            norm = np.ones(len(importances)) / len(importances)
        else:
            norm = np.clip(importances, 0, None) / total

        # Gini concentration index
        sorted_norm = np.sort(norm)
        n = len(sorted_norm)
        gini = float(
            (2 * np.sum((np.arange(1, n + 1)) * sorted_norm) - (n + 1) * sorted_norm.sum())
            / (n * sorted_norm.sum() + 1e-9)
        )

        # Top-k features that explain 80 % of importance
        cumulative = np.cumsum(np.sort(norm)[::-1])
        k80 = int(np.searchsorted(cumulative, 0.80)) + 1

        top_features: List[Dict] = sorted(
            [{"feature_index": int(i), "importance": round(float(importances[i]), 5),
              "std": round(float(importances_std[i]), 5)}
             for i in range(len(importances))],
            key=lambda d: -d["importance"],
        )[:20]

        # Max single-feature drop as fraction of base score (leakage proxy)
        max_drop_pct = float(np.max(importances) / (abs(float(importances.sum())) + 1e-9))

        # Score: penalise high concentration and potential leakage
        concentration_penalty = gini * 0.4
        leakage_penalty = min(0.5, max(0.0, max_drop_pct - 0.5) * 0.6)
        score = max(0.0, 1.0 - concentration_penalty - leakage_penalty)

        suggestions = []
        if gini > 0.70:
            suggestions.append(f"High feature concentration (Gini={gini:.2f}) — model relies heavily on few features, which may be brittle.")
        if k80 == 1:
            suggestions.append("A single feature drives 80 %+ of predictions — check for data leakage.")
        if max_drop_pct > 0.80:
            suggestions.append("One feature dominates importance. Verify it is not a target proxy.")

        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score, warn_below=0.70, fail_below=0.50),
            summary=(
                f"Gini concentration {gini:.2f} | "
                f"Features for 80% importance: {k80} / {X.shape[1]} | "
                f"Top-1 importance share: {max_drop_pct:.1%}"
            ),
            details={
                "gini_concentration": round(gini, 4),
                "features_for_80pct": k80,
                "top_features": top_features,
                "max_single_feature_share": round(max_drop_pct, 4),
            },
            suggestions=suggestions,
        )
