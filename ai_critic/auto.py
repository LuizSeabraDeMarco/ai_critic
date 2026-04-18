def detect_problem_type(y):
    unique = set(y)
    if len(unique) < 20:
        return "classification"
    return "regression"


def auto_configure(model, X, y):
    config = {}

    config["problem_type"] = detect_problem_type(y)

    if hasattr(model, "predict_proba"):
        config["supports_proba"] = True

    return config