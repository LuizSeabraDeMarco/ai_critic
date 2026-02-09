# ai_critic/ml/suggester.py

def suggest_fix(event: dict) -> dict:
    """
    Lightweight ML-ready suggestion engine.
    Today: rule-based.
    Tomorrow: trained on global telemetry.
    """

    signals = event["signals"]
    score = event["score"]

    # 🔴 Casos críticos
    if signals["leakage"] and signals["perfect_cv"]:
        return {
            "verdict": "critical",
            "suggestion": (
                "Strong evidence of data leakage. "
                "Audit features highly correlated with the target, "
                "remove shortcuts and re-run validation."
            )
        }

    # 🟠 Robustez fraca
    if signals["robustness"] == "fragile":
        return {
            "verdict": "warning",
            "suggestion": (
                "Model is fragile under noise. "
                "Consider stronger regularization, "
                "simpler architecture or more data."
            )
        }

    # 🟠 Estrutura pesada
    if signals["structural"] == "high":
        return {
            "verdict": "warning",
            "suggestion": (
                "Model complexity may be too high for dataset size. "
                "Reduce depth, number of parameters or features."
            )
        }

    # 🟢 Caso saudável
    if score >= 85:
        return {
            "verdict": "ok",
            "suggestion": (
                "Model behavior looks consistent. "
                "No critical risks detected at this stage."
            )
        }

    # 🟡 Default
    return {
        "verdict": "review",
        "suggestion": (
            "No critical failures detected, "
            "but model could benefit from further validation "
            "and robustness checks."
        )
    }
