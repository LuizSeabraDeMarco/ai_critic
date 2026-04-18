from .core import AICritic
from .auto import auto_configure

def check(model, X, y):
    config = auto_configure(model, X, y)

    critic = AICritic()
    report = critic.evaluate(model, X, y)

    return {
        "risk": report["risk"]["global_score"],
        "verdict": report["risk"]["verdict"],
        "summary": report["summary"]
    }