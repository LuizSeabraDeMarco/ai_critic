from sklearn.model_selection import cross_val_score
import numpy as np

from ai_critic.plugins.base import EvaluatorPlugin
from ai_critic.plugins.registry import EvaluatorRegistry
from .validation import make_cv


class PerformanceEvaluator(EvaluatorPlugin):

    name = "performance"
    dependencies = []

    def evaluate(self, model, dataset, context=None):
        """
        Evaluate model performance using cross-validation.
        """

        X = dataset["X"]
        y = dataset["y"]

        cv = make_cv(y)

        scores = cross_val_score(model, X, y, cv=cv)

        mean = float(np.mean(scores))
        std = float(np.std(scores))

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
            ),
        }


# Auto-register plugin
EvaluatorRegistry.register(PerformanceEvaluator())