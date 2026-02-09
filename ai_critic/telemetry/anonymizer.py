def anonymize(report: dict) -> dict:
    return {
        "model_type": report["meta"]["model_type"],
        "score": report["score"]["global"],
        "signals": {
            "leakage": report["details"]["data"]["data_leakage"]["suspected"],
            "robustness": report["details"]["robustness"]["verdict"],
        }
    }
