from abc import ABC, abstractmethod
from core.types import EvaluationResult, CriticInput


class Evaluator(ABC):
    name = "base"

    @abstractmethod
    def evaluate(self, data: CriticInput) -> EvaluationResult:
        pass