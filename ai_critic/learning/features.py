def extract_features(report: dict) -> dict:
    data = report["details"]["data"]
    perf = report["details"]["performance"]
    robust = report["details"]["robustness"]
    config = report["details"]["config"]

    return {
        "n_samples": report["meta"]["n_samples"],
        "n_features": report["meta"]["n_features"],
        "data_leakage": int(data["data_leakage"]["suspected"]),
        "perfect_cv": int(perf["suspiciously_perfect"]),
        "robustness_fragile": int(robust["verdict"] == "fragile"),
        "robustness_misleading": int(robust["verdict"] == "misleading"),
        "structural_risk_high": int(config["risk_level"] == "high"),
    }
