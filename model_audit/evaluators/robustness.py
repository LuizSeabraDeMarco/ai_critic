"""
Robustness Evaluator
====================
Tests how much the model degrades under:
  1. Gaussian noise injection (multiple intensity levels)
  2. Feature dropout (random zeroing of features)
  3. Outlier injection (extreme values in a subset of samples)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from model_audit.core.base import BaseEvaluator
from model_audit.core.types import DimensionResult, ProblemType, Verdict


class RobustnessEvaluator(BaseEvaluator):

    name = "robustness"
    weight = 1.2
    depends_on = ["performance"]

    # Noise levels as fraction of feature std
    _NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20]

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        cv = self._make_cv(y)

        base_score = (
            context["performance"].details["means"].get("accuracy")
            or context["performance"].details["means"].get("r2")
            if context and "performance" in context
            else None
        )
        if base_score is None:
            base_score = float(cross_val_score(clone(model), X, y, cv=cv).mean())

        scale = np.std(X, axis=0) + 1e-8

        # ── 1. Noise curve ──────────────────────────────────────────
        noise_drops: Dict[str, float] = {}
        for level in self._NOISE_LEVELS:
            noise = np.random.normal(0, level * scale, X.shape)
            score = float(cross_val_score(clone(model), X + noise, y, cv=cv).mean())
            drop = float(base_score - score)
            noise_drops[f"noise_{int(level*100)}pct"] = round(drop, 4)

        max_noise_drop = max(noise_drops.values())

        # ── 2. Feature dropout (random 30 % of features zeroed) ────
        rng = np.random.default_rng(42)
        n_drop = max(1, int(X.shape[1] * 0.30))
        drop_idx = rng.choice(X.shape[1], size=n_drop, replace=False)
        X_dropped = X.copy()
        X_dropped[:, drop_idx] = 0.0
        dropout_score = float(cross_val_score(clone(model), X_dropped, y, cv=cv).mean())
        dropout_drop = float(base_score - dropout_score)

        # ── 3. Outlier injection (5 % of samples ← extreme values) ─
        n_outliers = max(1, int(X.shape[0] * 0.05))
        outlier_idx = rng.choice(X.shape[0], size=n_outliers, replace=False)
        X_outlier = X.copy()
        X_outlier[outlier_idx] = X_outlier[outlier_idx] * 10
        outlier_score = float(cross_val_score(clone(model), X_outlier, y, cv=cv).mean())
        outlier_drop = float(base_score - outlier_score)

        # ── Composite robustness score ──────────────────────────────
        robustness = max(
            0.0,
            1.0
            - max_noise_drop * 0.5
            - max(0.0, dropout_drop) * 0.3
            - max(0.0, outlier_drop) * 0.2,
        )

        suggestions = []
        if max_noise_drop > 0.10:
            suggestions.append("Model is sensitive to noise. Try regularization or ensemble methods.")
        if dropout_drop > 0.15:
            suggestions.append("Model relies heavily on a subset of features. Consider feature selection.")
        if outlier_drop > 0.15:
            suggestions.append("Model is brittle to outliers. Consider robust scalers or outlier removal.")

        verdict = self._score_to_verdict(robustness, warn_below=0.80, fail_below=0.60)

        return DimensionResult(
            name=self.name,
            score=robustness,
            verdict=verdict,
            summary=(
                f"Noise drop (max) {max_noise_drop:.3f} | "
                f"Dropout drop {dropout_drop:.3f} | "
                f"Outlier drop {outlier_drop:.3f}"
            ),
            details={
                "base_score": round(base_score, 4),
                "noise_drops": noise_drops,
                "dropout_drop": round(dropout_drop, 4),
                "outlier_drop": round(outlier_drop, 4),
            },
            suggestions=suggestions,
        )
