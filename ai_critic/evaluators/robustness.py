import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional

from ai_critic.plugins.base import EvaluatorPlugin
from ai_critic.plugins.registry import EvaluatorRegistry
from .validation import make_cv


class RobustnessEvaluator(EvaluatorPlugin):
    """
    Tests model stability under controlled noise injection.
    """
    name = "robustness"
    dependencies = ["performance"]
    weight = 0.8

    def evaluate(self, model: Any, dataset: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        X = dataset["X"]
        y = dataset["y"]

        # Use performance result if available in context
        perf_score = None
        if context and "performance" in context:
            perf_score = context["performance"].get("score")

        noise_level = 0.02
        scale = np.std(X)
        noise = np.random.normal(0, noise_level * scale, X.shape)
        X_noisy = X + noise

        cv = make_cv(y)

        # If we don't have performance score from context, calculate it
        if perf_score is None:
            perf_score = cross_val_score(clone(model), X, y, cv=cv).mean()

        score_noisy = cross_val_score(clone(model), X_noisy, y, cv=cv).mean()

        drop = float(perf_score - score_noisy)
        verdict = "fragile" if drop > 0.15 else "stable"
        robustness_score = max(0.0, 1.0 - drop)

        return {
            "score": float(robustness_score),
            "performance_drop": drop,
            "verdict": verdict,
            "message": f"Model is {verdict} with a performance drop of {drop:.4f} under noise."
        }

# Auto-register plugin
EvaluatorRegistry.register(RobustnessEvaluator())
