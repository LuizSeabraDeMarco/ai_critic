from ai_critic.critic import AICritic


def benchmark(models, X, y):
    """
    Compare multiple models using AI Critic evaluation.
    """

    critic = AICritic()

    results = []

    for model in models:

        evaluation = critic.evaluate(model, X, y)

        score = evaluation["scores"].get("overall", 0)

        results.append({
            "model": type(model).__name__,
            "score": score,
            "details": evaluation
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results