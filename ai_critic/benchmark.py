from typing import List, Any, Dict
from ai_critic.critic import AICritic


def benchmark(models: List[Any], X: Any, y: Any) -> List[Dict[str, Any]]:
    """
    Compare multiple models using AI Critic evaluation.
    """
    critic = AICritic()
    results = []

    for model in models:
        report = critic.evaluate(model, X, y)
        score = report.scores.get("overall", 0.0)

        results.append({
            "model": type(model).__name__,
            "score": float(score),
            "verdict": report.scores.get("verdict"),
            "report": report.to_dict(),
            "summary": report.summary()
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
