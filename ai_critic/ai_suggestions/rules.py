def suggest(report):
    if report["score"]["global"] < 60:
        return "Reduce model complexity or audit features for leakage."
