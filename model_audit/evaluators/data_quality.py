"""
Data Quality Evaluator
======================
Audits the dataset itself, independent of the model:
  - Missing values
  - Duplicate rows
  - Class imbalance (classification)
  - Target leakage via correlation
  - Outlier density (IQR method)
  - Feature variance (constant / near-constant features)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from model_audit.core.base import BaseEvaluator
from model_audit.core.types import DimensionResult, ProblemType, Verdict


class DataQualityEvaluator(BaseEvaluator):

    name = "data_quality"
    weight = 1.0

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        n, p = X.shape
        penalty = 0.0
        issues: List[str] = []
        details: Dict = {"n_samples": n, "n_features": p}

        # ── Missing values ──────────────────────────────────────────
        missing_mask = np.isnan(X)
        missing_rate = float(missing_mask.mean())
        details["missing_rate"] = round(missing_rate, 4)
        if missing_rate > 0:
            issues.append(f"Missing values in X: {missing_rate:.1%}")
            penalty += min(0.40, missing_rate * 2)

        # ── Duplicate rows ──────────────────────────────────────────
        try:
            unique_rows = len(np.unique(X, axis=0))
            dup_rate = 1.0 - unique_rows / n
        except Exception:
            dup_rate = 0.0
        details["duplicate_row_rate"] = round(dup_rate, 4)
        if dup_rate > 0.05:
            issues.append(f"Duplicate rows: {dup_rate:.1%}")
            penalty += min(0.20, dup_rate)

        # ── Constant / near-constant features ──────────────────────
        stds = np.nanstd(X, axis=0)
        constant_cols = int(np.sum(stds < 1e-8))
        near_constant_cols = int(np.sum(stds < 0.01)) - constant_cols
        details["constant_features"] = constant_cols
        details["near_constant_features"] = near_constant_cols
        if constant_cols > 0:
            issues.append(f"{constant_cols} constant feature(s) detected — should be removed.")
            penalty += min(0.15, constant_cols / p * 0.5)

        # ── Outlier density (IQR method) ────────────────────────────
        Q1 = np.nanpercentile(X, 25, axis=0)
        Q3 = np.nanpercentile(X, 75, axis=0)
        IQR = Q3 - Q1 + 1e-8
        outlier_mask = ((X < Q1 - 3 * IQR) | (X > Q3 + 3 * IQR))
        outlier_rate = float(outlier_mask.any(axis=1).mean())
        details["outlier_row_rate"] = round(outlier_rate, 4)
        if outlier_rate > 0.10:
            issues.append(f"High outlier density: {outlier_rate:.1%} of rows have extreme values.")
            penalty += min(0.15, outlier_rate * 0.5)

        # ── Class imbalance (classification only) ──────────────────
        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            classes, counts = np.unique(y, return_counts=True)
            imbalance_ratio = float(counts.min() / counts.max())
            details["class_imbalance_ratio"] = round(imbalance_ratio, 4)
            details["class_distribution"] = {int(c): int(cnt) for c, cnt in zip(classes, counts)}
            if imbalance_ratio < 0.20:
                issues.append(f"Severe class imbalance (ratio {imbalance_ratio:.2f}). Consider oversampling or class weights.")
                penalty += 0.20
            elif imbalance_ratio < 0.50:
                issues.append(f"Moderate class imbalance (ratio {imbalance_ratio:.2f}).")
                penalty += 0.10

        # ── Target leakage via Pearson correlation ──────────────────
        y_float = y.astype(float)
        leaking_features = []
        for i in range(p):
            col = X[:, i]
            if np.std(col) < 1e-8:
                continue
            corr = float(np.corrcoef(col, y_float)[0, 1])
            if abs(corr) > 0.97:
                leaking_features.append({"feature_index": i, "correlation": round(corr, 4)})
        details["suspected_leakage_features"] = leaking_features
        if leaking_features:
            issues.append(f"{len(leaking_features)} feature(s) with |corr| > 0.97 with target — possible leakage.")
            penalty += min(0.50, len(leaking_features) * 0.15)

        score = max(0.0, 1.0 - penalty)
        details["issues"] = issues

        suggestions = []
        if missing_rate > 0:
            suggestions.append("Impute or drop missing values before training.")
        if leaking_features:
            suggestions.append("Investigate features with near-perfect target correlation.")
        if constant_cols > 0:
            suggestions.append("Remove constant features — they add no predictive signal.")
        if outlier_rate > 0.10:
            suggestions.append("Apply RobustScaler or clip extreme values.")

        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score, warn_below=0.80, fail_below=0.60),
            summary=(
                f"Missing {missing_rate:.1%} | "
                f"Duplicates {dup_rate:.1%} | "
                f"Outlier rows {outlier_rate:.1%} | "
                f"Leakage suspects {len(leaking_features)}"
            ),
            details=details,
            suggestions=suggestions,
        )
