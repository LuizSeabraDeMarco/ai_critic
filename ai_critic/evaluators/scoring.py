def compute_scores(report: dict) -> dict:

    details = report.get("details", {})

    data = details.get("data", {})
    perf = details.get("performance", {})
    robustness = details.get("robustness", {})
    explain = details.get("explainability", {})
    config = details.get("config", {})

    scores = {
        "data_integrity": 100,
        "validation": 100,
        "robustness": 100,
        "explainability": 100,
        "structure": 100
    }

    penalties = []

    # DATA
    if data.get("data_leakage", {}).get("suspected"):
        scores["data_integrity"] -= 50
        penalties.append("Data leakage detected")

    # VALIDATION
    if perf.get("suspiciously_perfect"):
        scores["validation"] -= 35
        penalties.append("Suspicious CV score")

    # ROBUSTNESS
    if robustness.get("verdict") == "fragile":
        scores["robustness"] -= 35

    # EXPLAINABILITY
    if explain.get("verdict") == "feature_leakage_risk":
        scores["explainability"] -= 40

    # STRUCTURE
    if config.get("risk_level") == "high":
        scores["structure"] -= 30

    # normalize
    for k in scores:
        scores[k] = max(0, min(100, scores[k]))

    weights = {
        "data_integrity": 0.3,
        "validation": 0.2,
        "robustness": 0.2,
        "explainability": 0.2,
        "structure": 0.1
    }

    global_score = round(sum(scores[k] * weights[k] for k in scores), 1)

    verdict = (
        "reliable" if global_score >= 85 else
        "usable_with_caution" if global_score >= 65 else
        "high_risk"
    )

    return {
        "global_score": global_score,
        "verdict": verdict,
        "component_scores": scores,
        "penalties": penalties
    }