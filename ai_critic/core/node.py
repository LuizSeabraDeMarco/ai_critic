from abc import ABC, abstractmethod
from typing import Dict, Any, List


class EvaluationNode(ABC):
    name: str = ""
    dependencies: List[str] = []

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
