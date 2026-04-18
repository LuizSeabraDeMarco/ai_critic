def enforce_minimum_quality(report, threshold=70):
    score = report["risk"]["global_score"]

    if score < threshold:
        raise Exception(
            f"Model rejected. Risk score too low: {score}"
        )