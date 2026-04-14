from evaluators.base import Evaluator
from core.types import EvaluationResult, CriticInput
from evaluators.registry import register


class AccuracyEvaluator(Evaluator):
    name = "accuracy"

    def evaluate(self, data: CriticInput) -> EvaluationResult:
        score = 1.0 if data.input_data == data.model_output else 0.0

        return EvaluationResult(
            name=self.name,
            score=score,
            explanation="Exact match comparison",
            confidence=0.9
        )


# registra automaticamente
register(AccuracyEvaluator)