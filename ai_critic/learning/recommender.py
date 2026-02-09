def recommend_changes(report):
    recs = []

    config = report["details"]["config"]
    perf = report["details"]["performance"]
    data = report["details"]["data"]

    if config["risk_level"] == "high":
        recs.append(
            "Reduce model complexity (e.g., lower max_depth, fewer estimators)."
        )

    if perf["suspiciously_perfect"]:
        recs.append(
            "Suspiciously perfect performance detected — verify data leakage."
        )

    if data["data_leakage"]["suspected"]:
        recs.append(
            "Potential target leakage — review feature engineering pipeline."
        )

    if not recs:
        recs.append("No critical changes recommended.")

    return recs
