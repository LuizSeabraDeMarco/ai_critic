import json
import sys

from ai_critic.telemetry.schema import CriticScorecard
from ai_critic.critic import AICritic


def run_gate(
    model,
    X,
    y,
    framework="sklearn",
    fail_on_risk="medium"
):
    critic = AICritic(
        model=model,
        X=X,
        y=y,
        framework=framework
    )

    report = critic.evaluate(view="all")
    scores = report["scores"]

    risk = scores["risk_level"]
    verdict = scores["verdict"]

    # Regra de falha
    should_fail = (
        verdict == "BLOCK"
        or (fail_on_risk == "medium" and risk in ("medium", "high"))
        or (fail_on_risk == "high" and risk == "high")
    )

    scorecard = {
        "critic_version": "2.1",
        "model_type": report["meta"]["framework"],
        "framework": report["meta"]["framework"],
        "n_samples": report["meta"]["n_samples"],
        "n_features": report["meta"]["n_features"],
        "score": scores["global"],
        "verdict": verdict,
        "risk_level": risk,
        "block_deploy": should_fail,
        "signals": {
            "leakage": report["details"]["data"]["data_leakage"]["suspected"],
            "perfect_cv": report["details"]["performance"]["suspiciously_perfect"],
            "robustness": report["details"]["robustness"]["verdict"],
            "structural_risk": report["details"]["config"]["risk_level"]
        },
        "recommended_actions": report["technical"]["recommendations"]
    }

    print(json.dumps(scorecard, indent=2))

    if should_fail:
        print("\n❌ AI CRITIC BLOCKED DEPLOYMENT", file=sys.stderr)
        sys.exit(1)

    print("\n✅ AI CRITIC PASSED")
    sys.exit(0)
