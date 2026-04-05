from typing import Dict, Any


def build_report(
    raw_results: Dict[str, Any],
    scores: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Unified report schema across the entire system.
    """

    return {
        "scores": scores,           # base aggregator (0–1)
        "details": raw_results,     # node outputs
        "risk": {},                 # scoring.py (0–100)
        "summary": {},              # human readable
        "suggestions": []           # actionable insights
    }