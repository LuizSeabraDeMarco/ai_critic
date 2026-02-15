from ai_critic.core.graph import EvaluationGraph
from .core.aggregator import ScoreAggregator

from ai_critic.evaluators.performance import PerformanceEvaluator
from ai_critic.evaluators.robustness import RobustnessEvaluator
from ai_critic.evaluators.explainability import ExplainabilityEvaluator


class AICritic:

    def __init__(self):
        self.graph = EvaluationGraph([
            PerformanceEvaluator(),
            RobustnessEvaluator(),
            ExplainabilityEvaluator(),
        ])

        self.aggregator = ScoreAggregator()

    def evaluate(self, model, X, y):

        results = self.graph.run({
            "model": model,
            "X": X,
            "y": y
        })

        scores = self.aggregator.aggregate(results)

        return {
            "scores": scores,
            "details": results
        }
