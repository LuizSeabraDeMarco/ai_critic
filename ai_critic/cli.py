import argparse
import json
import sys

from api.client import AICritic


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ai-critic",
        description="AI Critic — Production Readiness Check"
    )

    parser.add_argument("--model", required=True, help="Path to model (.pkl/.joblib)")
    parser.add_argument("--data", required=True, help="Path to dataset (.csv or .npz)")
    parser.add_argument("--target", required=True, help="Target column name")

    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--threshold", type=float, default=70.0, help="Minimum risk score to pass")
    parser.add_argument("--fail-on-risk", action="store_true", help="Fail if risk < threshold")

    return parser


def load_data(path, target, limit=None):
    import pandas as pd
    import numpy as np

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        data = np.load(path)
        df = pd.DataFrame(data["X"])
        df[target] = data["y"]

    if limit:
        df = df.head(limit)

    X = df.drop(columns=[target]).values
    y = df[target].values

    return X, y


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        import joblib

        # 📦 Load model
        model = joblib.load(args.model)

        # 📊 Load data
        X, y = load_data(args.data, args.target, args.limit)

        # 🤖 Predictions
        predictions = model.predict(X)

        # 🧠 Run critic
        critic = AICritic()
        report = critic.evaluate(
            input_data=y,
            model_output=predictions
        )

        # 🔥 Normalize output (NEW STANDARD FORMAT)
        risk_score = report.get("risk", {}).get("global_score", None)
        verdict = report.get("risk", {}).get("verdict", "unknown")

        output = {
            "risk_score": risk_score,
            "verdict": verdict,
            "details": report.get("details", {}),
            "summary": report.get("summary", {})
        }

        # 📤 Output
        if args.json:
            print(json.dumps(output, indent=2, default=str))
        else:
            print("\n=== AI CRITIC REPORT ===\n")
            print(f"Risk Score: {risk_score}")
            print(f"Verdict: {verdict}\n")

            if "summary" in output:
                for k, v in output["summary"].items():
                    print(f"{k}: {v}")

        # 🚫 QUALITY GATE (CRUCIAL PRA VIRAR PADRÃO)
        if args.fail_on_risk:
            if risk_score is not None and risk_score < args.threshold:
                print(
                    f"\n[ai-critic] ❌ Model rejected (risk {risk_score} < {args.threshold})",
                    file=sys.stderr
                )
                sys.exit(2)

    except Exception as e:
        print(f"[ai-critic] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()