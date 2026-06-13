"""
Complexity Evaluator
====================
Audits the model's structural complexity and inference cost:
  - Number of parameters / estimators
  - Tree depth heuristics (for tree-based models)
  - Feature-to-sample ratio risk
  - Inference latency estimate (median of 100 single-row predictions)
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from model_audit.core.base import BaseEvaluator
from model_audit.core.types import DimensionResult, ProblemType, Verdict


class ComplexityEvaluator(BaseEvaluator):

    name = "complexity"
    weight = 0.7

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: ProblemType,
        context: Optional[Dict[str, DimensionResult]] = None,
    ) -> DimensionResult:

        n, p = X.shape
        params = model.get_params() if hasattr(model, "get_params") else {}
        model_type = type(model).__name__

        details: Dict = {
            "model_type": model_type,
            "n_params_hyperparams": len(params),
            "uses_random_state": "random_state" in params,
        }

        penalty = 0.0
        issues = []

        # ── Tree depth heuristic ────────────────────────────────────
        if "max_depth" in params and params["max_depth"] is not None:
            import math
            recommended = math.log2(max(2, n))
            depth = params["max_depth"]
            details["max_depth"] = depth
            details["recommended_max_depth"] = round(recommended, 1)
            if depth > recommended * 2:
                issues.append(f"max_depth={depth} is very high for {n} samples (recommended ≤ {int(recommended)}).")
                penalty += 0.25
            elif depth > recommended:
                issues.append(f"max_depth={depth} exceeds recommended {int(recommended)} for dataset size.")
                penalty += 0.10

        # ── n_estimators ────────────────────────────────────────────
        if "n_estimators" in params:
            ne = params["n_estimators"]
            details["n_estimators"] = ne
            if ne > 1000:
                issues.append(f"n_estimators={ne} is very large — high inference cost.")
                penalty += 0.05

        # ── Feature / sample ratio ───────────────────────────────────
        ratio = p / n
        details["feature_sample_ratio"] = round(ratio, 4)
        if ratio > 1.0:
            issues.append("More features than samples — high risk of overfitting.")
            penalty += 0.30
        elif ratio > 0.5:
            issues.append("Feature / sample ratio > 0.5 — consider dimensionality reduction.")
            penalty += 0.10

        # ── Inference latency ────────────────────────────────────────
        try:
            fitted = model  # assumes model is already fitted by the time graph reaches here
            row = X[:1]
            latencies = []
            for _ in range(50):
                t0 = time.perf_counter()
                fitted.predict(row)
                latencies.append((time.perf_counter() - t0) * 1000)
            median_ms = float(np.median(latencies))
            details["inference_latency_ms_median"] = round(median_ms, 3)
            if median_ms > 100:
                issues.append(f"Median inference latency {median_ms:.1f} ms — may be too slow for real-time use.")
                penalty += 0.10
        except Exception:
            details["inference_latency_ms_median"] = None

        score = max(0.0, 1.0 - penalty)
        details["issues"] = issues

        suggestions = []
        if ratio > 1.0:
            suggestions.append("Apply PCA or feature selection to reduce dimensionality.")
        if "max_depth" in details and penalty > 0.15:
            suggestions.append("Limit tree depth to prevent overfitting.")

        return DimensionResult(
            name=self.name,
            score=score,
            verdict=self._score_to_verdict(score, warn_below=0.80, fail_below=0.60),
            summary=f"Model: {model_type} | Feature/sample ratio: {ratio:.2f} | Issues: {len(issues)}",
            details=details,
            suggestions=suggestions,
        )
