from typing import List, Dict, Any, Optional
from ai_critic.core.result import EvaluationReport


class SuggestionEngine:
    """
    Analyzes evaluation reports to provide actionable ML advice.
    """

    @staticmethod
    def suggest(report: EvaluationReport) -> List[Dict[str, str]]:
        """
        Generate a list of suggestions based on report results.
        """
        suggestions = []
        scores = report.scores
        details = report.details

        # Overall performance check
        overall = scores.get("overall", 0.0)
        if overall < 0.6:
            suggestions.append({
                "category": "performance",
                "priority": "HIGH",
                "message": "Low overall score. Consider feature engineering or a more complex model architecture."
            })

        # Robustness check
        robustness = details.get("robustness", {})
        if robustness.get("verdict") == "fragile":
            suggestions.append({
                "category": "robustness",
                "priority": "HIGH",
                "message": "Model is sensitive to noise. Try regularization (L1/L2) or data augmentation."
            })

        # Explainability check
        explainability = details.get("explainability", {})
        if explainability.get("verdict") == "feature_leakage_risk":
            suggestions.append({
                "category": "leakage",
                "priority": "CRITICAL",
                "message": "Potential data leakage detected. Audit your features for target information leakage."
            })
        elif explainability.get("verdict") == "feature_dependency":
            suggestions.append({
                "category": "explainability",
                "priority": "MEDIUM",
                "message": "Model relies heavily on a few features. Verify if these features are stable in production."
            })

        # Performance consistency
        perf = details.get("performance", {})
        if perf.get("suspiciously_perfect"):
            suggestions.append({
                "category": "leakage",
                "priority": "CRITICAL",
                "message": "Suspiciously perfect performance. Check for label leakage or training/test overlap."
            })
        
        if perf.get("cv_std", 0) > 0.1:
            suggestions.append({
                "category": "stability",
                "priority": "MEDIUM",
                "message": "High variance in cross-validation. Consider more data or simpler model to reduce variance."
            })

        if not suggestions:
            suggestions.append({
                "category": "general",
                "priority": "LOW",
                "message": "Model looks solid across all dimensions. Monitor production performance."
            })

        return suggestions


def suggest(report: Any) -> List[Dict[str, str]]:
    """
    Legacy wrapper for SuggestionEngine.
    """
    if isinstance(report, dict):
        # Convert dict back to EvaluationReport if needed
        report_obj = EvaluationReport(
            scores=report.get("scores", {}),
            details=report.get("details", {}),
            metadata=report.get("metadata", {})
        )
    else:
        report_obj = report
        
    return SuggestionEngine.suggest(report_obj)
