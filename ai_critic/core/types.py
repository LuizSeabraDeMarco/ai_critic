from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class EvaluationResult:
    name: str
    score: float
    explanation: str
    confidence: float


@dataclass
class CriticInput:
    input_data: Any
    model_output: Any
    metadata: Dict = None