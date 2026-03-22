import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from typing import Dict, Any, Optional

from ai_critic.plugins.base import EvaluatorPlugin
from ai_critic.plugins.registry import EvaluatorRegistry
from .validation import make_cv


class ExplainabilityEvaluator(EvaluatorPlugin):
    """
    Uses permutation sensitivity analysis to estimate feature importance behavior.
    """
    name = "explainability"
    dependencies = ["performance"]
    weight = 0.7

    def evaluate(self, model: Any, dataset: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        X = dataset["X"]
        y = dataset["y"]

        cv = make_cv(y)

        # Use performance result if available in context
        base_score = None
        if context and "performance" in context:
            base_score = context["performance"].get("score")

        if base_score is None:
            base_score = cross_val_score(clone(model), X, y, cv=cv).mean()

        max_drop = 0.0
        
        # Limit the number of features to permute if X is large
        num_features = min(X.shape[1], 10)
        feature_indices = np.random.choice(X.shape[1], num_features, replace=False)

        for i in feature_indices:
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            score = cross_val_score(clone(model), X_permuted, y, cv=cv).mean()
            drop = base_score - score
            max_drop = max(max_drop, drop)

        if max_drop > 0.30:
            verdict = "feature_leakage_risk"
        elif max_drop > 0.15:
            verdict = "feature_dependency"
        else:
            verdict = "stable"

        explainability_score = max(0.0, 1.0 - max_drop)

        return {
            "score": float(explainability_score),
            "max_performance_drop": float(max_drop),
            "verdict": verdict,
            "message": f"Explainability check: {verdict} with max drop of {max_drop:.4f}."
        }

# Auto-register plugin
EvaluatorRegistry.register(ExplainabilityEvaluator())
