from sklearn.model_selection import cross_val_score
import numpy as np

from ai_critic.core.node import EvaluationNode
from .validation import make_cv


class PerformanceEvaluator(EvaluationNode):

    name = "performance"
    dependencies = []

    def evaluate(self, context):
        model = context["input"]["model"]
        X = context["input"]["X"]
        y = context["input"]["y"]

        cv = make_cv(y)

        scores = cross_val_score(model, X, y, cv=cv)
        mean = float(scores.mean())
        std = float(scores.std())
        suspicious = mean > 0.995

        return {
            "score": mean,
            "cv_mean_score": mean,
            "cv_std": std,
            "suspiciously_perfect": suspicious,
            "validation_strategy": type(cv).__name__,
            "message": (
                "Perfect CV score detected — possible data leakage."
                if suspicious
                else "CV performance within expected range."
            )
        }
