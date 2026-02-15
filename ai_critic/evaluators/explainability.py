import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

from ai_critic.core.node import EvaluationNode
from .validation import make_cv


class ExplainabilityEvaluator(EvaluationNode):

    name = "explainability"
    dependencies = ["performance"]

    def evaluate(self, context):
        model = context["input"]["model"]
        X = context["input"]["X"]
        y = context["input"]["y"]

        cv = make_cv(y)

        base_score = cross_val_score(
            clone(model), X, y, cv=cv
        ).mean()

        max_drop = 0.0

        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            score = cross_val_score(
                clone(model), X_permuted, y, cv=cv
            ).mean()

            drop = base_score - score
            max_drop = max(max_drop, drop)

        if max_drop > 0.30:
            verdict = "feature_leakage_risk"
        elif max_drop > 0.15:
            verdict = "feature_dependency"
        else:
            verdict = "stable"

        explainability_score = max(0.0, 1 - max_drop)

        return {
            "score": float(explainability_score),
            "max_performance_drop": float(max_drop),
            "verdict": verdict
        }
