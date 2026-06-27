"""
Calibration Evaluator  (classification only)
============================================
A well-calibrated model's predicted probability P(y=1|x) ≈ empirical
positive rate in that probability bucket.

Metrics
-------
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagram data (for visualisation)
- Brier Score (proper scoring rule)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from ai_critic.core.base import BaseEvaluator
from ai_critic.core.types import DimensionResult, ProblemType, Verdict


class CalibrationEvaluator(BaseEvaluator):

    name = "calibration"
    weight = 0.8
    depends_on = ["performance"]

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        if problem_type not in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            return DimensionResult(
                name=self.name,
                score=1.0,
                verdict=Verdict.PASS,
                summary="Calibration only applies to classifiers — skipped for regression.",
                details={"skipped": True},
            )

        if not hasattr(model, "predict_proba"):
            return DimensionResult(
                name=self.name,
                score=0.5,
                verdict=Verdict.WARNING,
                summary="Model does not support predict_proba — calibration cannot be measured.",
                details={"skipped": True},
                suggestions=["Wrap the model with CalibratedClassifierCV to enable probability calibration."],
            )

        # For binary: use positive-class probability
        # For multiclass: per-class calibration averaged
        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            return self._binary_calibration(model, X, y)
        return self._multiclass_calibration(model, X, y)

    # ------------------------------------------------------------------
    def _binary_calibration(self, model, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        probs_all, y_all = [], []

        for train_idx, val_idx in cv.split(X, y):
            m = clone(model).fit(X[train_idx], y[train_idx])
            probs_all.extend(m.predict_proba(X[val_idx])[:, 1].tolist())
            y_all.extend(y[val_idx].tolist())

        probs = np.array(probs_all)
        y_val = np.array(y_all)

        ece, mce, bins = self._compute_ece(probs, y_val)

        # Brier score
        brier = float(np.mean((probs - y_val) ** 2))

        score = max(0.0, 1.0 - ece * 3)  # ECE 0.33 → score 0

        suggestions = []
        if ece > 0.10:
            suggestions.append(f"ECE = {ece:.3f} — model is poorly calibrated. Use CalibratedClassifierCV(cv=5, method='isotonic').")
        if brier > 0.25:
            suggestions.append(f"Brier score = {brier:.3f} — predicted probabilities are inaccurate.")

        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score, warn_below=0.80, fail_below=0.60),
            summary=f"ECE {ece:.3f} | MCE {mce:.3f} | Brier {brier:.3f}",
            details={"ece": round(ece, 4), "mce": round(mce, 4), "brier": round(brier, 4), "reliability_bins": bins},
            suggestions=suggestions,
        )

    def _multiclass_calibration(self, model, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        classes = np.unique(y)
        all_ece = []

        probs_all, y_all = [], []
        for train_idx, val_idx in cv.split(X, y):
            m = clone(model).fit(X[train_idx], y[train_idx])
            probs_all.append(m.predict_proba(X[val_idx]))
            y_all.extend(y[val_idx].tolist())

        probs = np.vstack(probs_all)
        y_val = np.array(y_all)

        for i, cls in enumerate(classes):
            y_bin = (y_val == cls).astype(int)
            ece, _, _ = self._compute_ece(probs[:, i], y_bin)
            all_ece.append(ece)

        mean_ece = float(np.mean(all_ece))
        score = max(0.0, 1.0 - mean_ece * 3)

        suggestions = []
        if mean_ece > 0.10:
            suggestions.append(f"Mean ECE = {mean_ece:.3f} across classes — consider probability calibration.")

        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score, warn_below=0.80, fail_below=0.60),
            summary=f"Mean ECE across classes: {mean_ece:.3f}",
            details={"mean_ece": round(mean_ece, 4), "per_class_ece": [round(e, 4) for e in all_ece]},
            suggestions=suggestions,
        )

    @staticmethod
    def _compute_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10):
        bins_data: List[Dict] = []
        ece = 0.0
        mce = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= low) & (probs < high)
            if mask.sum() == 0:
                continue
            avg_conf = float(probs[mask].mean())
            avg_acc = float(y[mask].mean())
            gap = abs(avg_conf - avg_acc)
            ece += gap * mask.sum() / len(probs)
            mce = max(mce, gap)
            bins_data.append({
                "bin_lower": round(low, 2),
                "bin_upper": round(high, 2),
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
                "n": int(mask.sum()),
            })

        return float(ece), float(mce), bins_data
