def policy_decision(rule_decision: dict, ml_score: float):
    if rule_decision["risk_level"] == "high":
        return {
            "deploy": False,
            "reason": "Blocked by rules",
            "ml_score": ml_score
        }

    if ml_score < 0.4:
        return {
            "deploy": False,
            "reason": "ML predicts failure",
            "ml_score": ml_score
        }

    return {
        "deploy": True,
        "reason": "Approved by ML + rules",
        "ml_score": ml_score
    }
