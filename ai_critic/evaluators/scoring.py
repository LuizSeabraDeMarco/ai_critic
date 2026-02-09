def compute_scores(report: dict) -> dict:
    """
    Converts critic signals into an interpretable 0–100 score.
    This score is heuristic and diagnostic — NOT a benchmark.
    """

    # ---------- INITIAL SCORES ----------
    scores = {
        "data_integrity": 100,
        "validation": 100,
        "robustness": 100,
        "explainability": 100,
        "structure": 100
    }

    penalties = []

    # ---------- DATA INTEGRITY ----------
    data = report["details"]["data"]
    if data["data_leakage"]["suspected"]:
        scores["data_integrity"] -= 50
        penalties.append("Severe data leakage risk detected")

    if data.get("target_overlap", False):
        scores["data_integrity"] -= 30
        penalties.append("Target appears partially encoded in features")

    # ---------- VALIDATION ----------
    perf = report["details"]["performance"]

    if perf["suspiciously_perfect"]:
        scores["validation"] -= 35
        penalties.append("Suspiciously perfect validation score")

    cv_gap = perf.get("train_test_gap", 0)
    if cv_gap > 0.15:
        scores["validation"] -= 20
        penalties.append(f"High train-test gap detected ({cv_gap:.2f})")

    # ---------- ROBUSTNESS ----------
    robustness = report["details"]["robustness"]["verdict"]

    if robustness == "fragile":
        scores["robustness"] -= 35
        penalties.append("Model fragile under perturbations")

    elif robustness == "misleading":
        scores["robustness"] -= 55
        penalties.append("Model behavior misleading under stress")

    # ---------- EXPLAINABILITY ----------
    explain = report["details"].get("explainability", {})

    explain_verdict = explain.get("verdict")
    max_drop = explain.get("max_performance_drop", 0)

    if explain_verdict == "feature_leakage_risk":
        scores["explainability"] -= 45
        penalties.append("Explainability suggests feature leakage")

    elif explain_verdict == "feature_dependency":
        scores["explainability"] -= 25
        penalties.append("Model overly dependent on few features")

    if max_drop > 0.4:
        scores["explainability"] -= 20
        penalties.append("High performance collapse after feature removal")

    # ---------- STRUCTURE ----------
    structural_warnings = report["details"]["config"]["structural_warnings"]
    if structural_warnings:
        scores["structure"] -= min(30, 10 * len(structural_warnings))
        penalties.append("Structural configuration warnings detected")

    # ---------- NORMALIZATION ----------
    for k in scores:
        scores[k] = max(0, min(100, scores[k]))

    # ---------- WEIGHTED GLOBAL SCORE ----------
    weights = {
        "data_integrity": 0.30,
        "validation": 0.20,
        "robustness": 0.20,
        "explainability": 0.20,
        "structure": 0.10
    }

    global_score = round(
        sum(scores[k] * weights[k] for k in scores), 1
    )

    # ---------- VERDICT ----------
    if global_score >= 85:
        verdict = "reliable"
    elif global_score >= 65:
        verdict = "usable_with_caution"
    else:
        verdict = "high_risk"

    return {
        "global_score": global_score,
        "verdict": verdict,
        "component_scores": scores,
        "penalties": penalties,
        "confidence_level": (
            "high" if global_score >= 80
            else "medium" if global_score >= 60
            else "low"
        )
    }
