import argparse
import json
import sys

from ai_critic.critic import AICritic


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ai-critic",
        description="AI Critic — Evaluation Graph Engine"
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--json", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
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

        X = df.drop(columns=[args.target]).values
        y = df[args.target].values

        critic = AICritic()
        report = critic.evaluate(model, X, y)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("\n=== AI CRITIC REPORT ===\n")
            print(f"Overall score: {report['scores']['overall']:.3f}")
            print(f"Verdict: {report['scores']['verdict']}\n")

    except Exception as e:
        print(f"[ai-critic] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
