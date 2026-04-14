from typing import List
from core.types import CriticInput, EvaluationResult
from evaluators.registry import get_evaluators


class CriticPipeline:
    def __init__(self, evaluators=None):
        self.evaluators = evaluators or get_evaluators()

    def run(self, data: CriticInput) -> List[EvaluationResult]:
        results = []

        for evaluator in self.evaluators:
            result = evaluator.evaluate(data)
            results.append(result)

        return results

    def aggregate(self, results: List[EvaluationResult]):
        if not results:
            return 0

        return sum(r.score for r in results) / len(results)