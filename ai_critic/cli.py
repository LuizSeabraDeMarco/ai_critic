import argparse
import json
import sys

from ai_critic.critic import AICritic
from ai_critic.learning.critic_gate import CriticGate
from ai_critic.telemetry.schema import TelemetryEvent


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ai-critic",
        description="AI Critic — Intelligent model evaluation & deployment advisor"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to a serialized model (pickle/joblib)"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset (npz or csv)"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--framework",
        default="sklearn",
        help="ML framework (default: sklearn)"
    )

    parser.add_argument(
        "--session",
        help="Session ID for tracking evaluations"
    )

    parser.add_argument(
        "--feedback",
        choices=["success", "fail"],
        help="Optional human feedback after deployment"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        # 🔹 Lazy imports (evita custo no CLI)
        import joblib
        import pandas as pd
        import numpy as np

        model = joblib.load(args.model)

        if args.data.endswith(".csv"):
            df = pd.read_csv(args.data)
        else:
            data = np.load(args.data)
            df = pd.DataFrame(data["X"])
            df[args.target] = data["y"]

        X = df.drop(columns=[args.target])
        y = df[args.target]

        critic = AICritic(
            model=model,
            X=X,
            y=y,
            framework=args.framework,
            session=args.session
        )

        report = critic.evaluate(view="all")

        # 🔹 Telemetry Event
        telemetry = TelemetryEvent(
            model_type=type(model).__name__,
            framework=args.framework,
            problem_type=report["details"]["performance"]["problem_type"],
            n_samples=report["meta"]["n_samples"],
            n_features=report["meta"]["n_features"],
            score=report["scores"]["overall"],
            verdict=report["scores"]["verdict"]
        )

        # 🔹 Gate decision
        gate = CriticGate()
        gate_decision = gate.decide(telemetry)

        # 🔹 Deploy decision (rule + ML)
        deploy_decision = critic.deploy_decision(
            success_feedback=args.feedback
        )

        output = {
            "gate": {
                "should_suggest": gate_decision.should_suggest,
                "confidence": gate_decision.confidence,
                "reason": gate_decision.reason
            },
            "deploy": deploy_decision,
        }

        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print("\n=== AI CRITIC REPORT ===\n")
            print(f"Score: {telemetry.score:.3f}")
            print(f"Verdict: {telemetry.verdict}")
            print(f"Risk level: {deploy_decision['risk_level']}\n")

            print("Gate decision:")
            print(f"  → Suggest improvements: {gate_decision.should_suggest}")
            print(f"  → Confidence: {gate_decision.confidence}")
            print(f"  → Reason: {gate_decision.reason}\n")

            print("Deployment:")
            print(f"  → Deploy: {deploy_decision['deploy']}")
            print(f"  → ML score: {deploy_decision['ml_score']}\n")

            if gate_decision.should_suggest:
                print("Recommendations:")
                for rec in deploy_decision["recommendations"]:
                    print(f"  • {rec}")

    except Exception as e:
        print(f"[ai-critic] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
