import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ai_critic.core.node import EvaluationNode
from .validation import make_cv


class RobustnessEvaluator(EvaluationNode):

    name = "robustness"
    dependencies = ["performance"]

    def evaluate(self, context):
        model = context["input"]["model"]
        X = context["input"]["X"]
        y = context["input"]["y"]

        noise_level = 0.02
        scale = np.std(X)
        noise = np.random.normal(0, noise_level * scale, X.shape)
        X_noisy = X + noise

        cv = make_cv(y)

        score_clean = cross_val_score(
            clone(model), X, y, cv=cv
        ).mean()

        score_noisy = cross_val_score(
            clone(model), X_noisy, y, cv=cv
        ).mean()

        drop = score_clean - score_noisy

        if drop > 0.15:
            verdict = "fragile"
        else:
            verdict = "stable"

        robustness_score = max(0.0, 1 - drop)

        return {
            "score": float(robustness_score),
            "performance_drop": float(drop),
            "verdict": verdict
        }
