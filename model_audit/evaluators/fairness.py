"""
Fairness Evaluator
==================
Measures whether the model performs equally well across different
demographic or categorical groups present in the dataset.

Strategy
--------
For each integer/boolean column that looks like a group indicator
(≤ 10 unique values, ≥ 2 groups, ≥ 20 samples per group), compute
per-group performance and measure:

  * Disparate Impact Ratio  (min_group_score / max_group_score)
  * Equalised Opportunity Gap  (max − min group score)

A score of 1.0 means perfectly equal performance. Score drops as
disparity grows.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone

from model_audit.core.base import BaseEvaluator
from model_audit.core.types import DimensionResult, ProblemType, Verdict


class FairnessEvaluator(BaseEvaluator):

    name = "fairness"
    weight = 1.1
    depends_on = ["performance"]

    _MAX_GROUPS = 10
    _MIN_GROUP_SIZE = 20

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        candidate_cols = self._find_group_columns(X)

        if not candidate_cols:
            return DimensionResult(
                name=self.name,
                score=1.0,
                verdict=Verdict.PASS,
                summary="No categorical group columns detected — fairness check skipped.",
                details={"skipped": True},
                suggestions=["Provide a pandas DataFrame with labelled columns for a richer fairness audit."],
            )

        group_results: Dict[int, Dict] = {}
        max_gap = 0.0
        min_dir = 1.0

        for col in candidate_cols:
            groups = np.unique(X[:, col])
            group_scores = {}
            for g in groups:
                mask = X[:, col] == g
                if mask.sum() < self._MIN_GROUP_SIZE:
                    continue
                X_g, y_g = X[mask], y[mask]
                try:
                    m = clone(model).fit(X[~mask], y[~mask])
                    preds = m.predict(X_g)
                    if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
                        from sklearn.metrics import f1_score
                        sc = float(f1_score(y_g, preds, average="macro", zero_division=0))
                    else:
                        from sklearn.metrics import r2_score
                        sc = float(r2_score(y_g, preds))
                    group_scores[int(g)] = round(sc, 4)
                except Exception:
                    pass

            if len(group_scores) >= 2:
                vals = list(group_scores.values())
                gap = max(vals) - min(vals)
                dir_ratio = min(vals) / max(vals) if max(vals) > 0 else 1.0
                max_gap = max(max_gap, gap)
                min_dir = min(min_dir, dir_ratio)
                group_results[col] = {"group_scores": group_scores, "gap": round(gap, 4), "dir_ratio": round(dir_ratio, 4)}

        if not group_results:
            return DimensionResult(
                name=self.name,
                score=1.0,
                verdict=Verdict.PASS,
                summary="Groups found but too small to measure — fairness check inconclusive.",
                details={"skipped": True},
            )

        # Score: 1 - max gap (clipped)
        fairness_score = max(0.0, 1.0 - max_gap)

        suggestions = []
        if max_gap > 0.20:
            suggestions.append(f"Performance gap of {max_gap:.2f} detected across groups — consider re-weighting or fairness constraints.")
        if min_dir < 0.80:
            suggestions.append(f"Disparate Impact Ratio {min_dir:.2f} < 0.80 — model may systematically underserve a group.")

        return DimensionResult(
            name=self.name,
            score=fairness_score,
            verdict=self._score_to_verdict(fairness_score, warn_below=0.85, fail_below=0.70),
            summary=f"Max performance gap across groups: {max_gap:.3f} | Min disparate-impact ratio: {min_dir:.3f}",
            details={"group_results": group_results, "max_gap": round(max_gap, 4), "min_dir_ratio": round(min_dir, 4)},
            suggestions=suggestions,
        )

    def _find_group_columns(self, X: np.ndarray) -> List[int]:
        candidates = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if not np.issubdtype(col.dtype, np.integer) and not np.issubdtype(col.dtype, np.bool_):
                # Accept float columns with integer-like values
                if not np.allclose(col, col.astype(int)):
                    continue
                col = col.astype(int)
            unique = np.unique(col)
            if 2 <= len(unique) <= self._MAX_GROUPS:
                if all(np.sum(col == u) >= self._MIN_GROUP_SIZE for u in unique):
                    candidates.append(i)
        return candidates
