import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class EvaluationReport:
    """
    Structured container for evaluation results.
    Provides methods for serialization and analysis.
    """

    def __init__(self, scores: Dict[str, Any], details: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        self.timestamp = datetime.now().isoformat()
        self.scores = scores
        self.details = details
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to a plain dictionary.
        """
        return {
            "timestamp": self.timestamp,
            "scores": self.scores,
            "details": self.details,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 4) -> str:
        """
        Serialize report to JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """
        Generate a human-readable text summary of the report.
        """
        overall = self.scores.get("overall", 0.0)
        verdict = self.scores.get("verdict", "unknown").upper()
        
        lines = [
            "=" * 40,
            "        AI CRITIC EVALUATION REPORT",
            "=" * 40,
            f"Timestamp: {self.timestamp}",
            f"Overall Score: {overall:.3f}",
            f"Verdict: {verdict}",
            "-" * 40,
            "Node Details:"
        ]

        for name, detail in self.details.items():
            score = detail.get("score", 0.0)
            msg = detail.get("message", "No message")
            lines.append(f" - {name.capitalize()}: {score:.3f} ({msg})")

        critical = self.scores.get("critical_issues", [])
        if critical:
            lines.append("-" * 40)
            lines.append("CRITICAL ISSUES DETECTED:")
            for issue in critical:
                lines.append(f" [!] {issue}")

        lines.append("=" * 40)
        return "\n".join(lines)

    def __repr__(self):
        return f"<EvaluationReport: {self.scores.get('verdict', 'N/A')} ({self.scores.get('overall', 0.0)})>"
