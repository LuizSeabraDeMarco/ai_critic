def build_event(report: dict) -> dict:
    return {
        "model_type": report["meta"]["model_type"],
        "framework": report["meta"]["framework"],
        "n_samples": report["meta"]["n_samples"],
        "n_features": report["meta"]["n_features"],
        "score": report["scores"]["global"],
        "risk_level": report["executive"]["risk_level"],
        "signals": {
            "leakage": report["details"]["data"]["data_leakage"]["suspected"],
            "perfect_cv": report["details"]["performance"]["suspiciously_perfect"],
            "robustness": report["details"]["robustness"]["verdict"],
            "structural": report["details"]["config"]["risk_level"],
        }
    }
