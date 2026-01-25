def compute_scores(report: dict) -> dict:
    """
    Converts critic signals into a coarse 0–100 score.
    Score is NOT an objective metric.
    """

    score = 100

    data_leakage = report["details"]["data"]["data_leakage"]["suspected"]
    perfect_cv = report["details"]["performance"]["suspiciously_perfect"]
    robustness = report["details"]["robustness"]["verdict"]
    structural = report["details"]["config"]["structural_warnings"]

    if data_leakage:
        score -= 30

    if perfect_cv:
        score -= 20

    if robustness == "fragile":
        score -= 15
    elif robustness == "misleading":
        score -= 25

    if structural:
        score -= 10

    return {
        "global": max(0, min(100, score)),
        "components": {
            "data_integrity": 0 if data_leakage else 100,
            "validation": 70 if perfect_cv else 100,
            "robustness": {
                "stable": 100,
                "fragile": 65,
                "misleading": 40
            }.get(robustness, 100),
        }
    }
