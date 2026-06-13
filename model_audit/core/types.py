"""
Core data types shared across all evaluators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class DimensionResult:
    """Result from a single evaluation dimension."""
    name: str
    score: float          # 0.0 – 1.0
    verdict: Verdict
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "verdict": self.verdict.value,
            "summary": self.summary,
            "details": self.details,
            "suggestions": self.suggestions,
        }


@dataclass
class AuditReport:
    """Unified report produced by ModelAudit."""
    problem_type: ProblemType
    dimensions: Dict[str, DimensionResult] = field(default_factory=dict)
    overall_score: float = 0.0
    overall_verdict: Verdict = Verdict.FAIL
    top_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type.value,
            "overall_score": round(self.overall_score, 4),
            "overall_verdict": self.overall_verdict.value,
            "top_suggestions": self.top_suggestions,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
        }
