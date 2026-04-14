import argparse
import json
import sys

from api.client import AICritic


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ai-critic",
        description="AI Critic — Evaluation Pipeline"
    )

    parser.add_argument("--model", required=True, help="Path to model (.pkl/.joblib)")
    parser.add_argument("--data", required=True, help="Path to dataset (.csv or .npz)")
    parser.add_argument("--target", required=True, help="Target column name")
    
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--limit", type=int, help="Limit number of samples (debug)")
    
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

        # 📦 carregar modelo
        model = joblib.load(args.model)

        # 📊 carregar dados
        X, y = load_data(args.data, args.target, args.limit)

        # 🤖 gerar previsões
        predictions = model.predict(X)

        # 🧠 rodar critic
        critic = AICritic()
        report = critic.evaluate(
            input_data=y,
            model_output=predictions
        )

        # 📤 saída
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("\n=== AI CRITIC REPORT ===\n")
            print(f"Final Score: {report['final_score']:.4f}\n")

            for r in report["details"]:
                print(f"[{r.name}]")
                print(f"  Score: {r.score:.4f}")
                print(f"  Confidence: {r.confidence:.2f}")
                print(f"  → {r.explanation}\n")

    except Exception as e:
        print(f"[ai-critic] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()