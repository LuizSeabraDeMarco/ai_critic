from typing import Dict, Any, List, Optional


class ScoreAggregator:
    """
    Aggregates individual evaluation results into a unified score.
    Supports weighted averages and metadata analysis.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {}

    def aggregate(self, results: Dict[str, Any], nodes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall score based on weights and individual scores.
        """
        if not results:
            return {
                "overall": 0.0,
                "verdict": "insufficient_data",
                "message": "No evaluation results provided."
            }

        total_score = 0.0
        total_weight = 0.0

        for name, result in results.items():
            if "score" in result:
                # Use node weight if defined, otherwise use provided weight or default to 1.0
                node = nodes.get(name)
                weight = self.weights.get(name, getattr(node, "weight", 1.0))
                
                total_score += result["score"] * weight
                total_weight += weight

        if total_weight == 0:
            overall = 0.0
        else:
            overall = round(total_score / total_weight, 3)

        verdict = self._get_verdict(overall)
        
        # Analyze critical failures
        critical_issues = [
            res.get("message") for res in results.values() 
            if res.get("suspiciously_perfect") or res.get("verdict") == "fragile"
        ]

        return {
            "overall": float(overall),
            "verdict": verdict,
            "critical_issues": critical_issues,
            "weighted": total_weight > len(results)  # Indicate if non-default weights were used
        }

    def _get_verdict(self, score: float) -> str:
        """
        Map numerical score to qualitative verdict.
        """
        if score >= 0.90:
            return "excellent"
        elif score >= 0.75:
            return "good"
        elif score >= 0.60:
            return "moderate"
        elif score >= 0.40:
            return "poor"
        else:
            return "unacceptable"
