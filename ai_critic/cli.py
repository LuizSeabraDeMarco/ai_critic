"""
ai-critic CLI
===============
Usage
-----
    ai-critic --help
    ai-critic score path/to/model.pkl path/to/data.csv --target col_name
    ai-critic score model.pkl data.csv --target y --report audit.html
    ai-critic score model.pkl data.csv --target y --gate 0.75
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_score(args) -> int:
    """Load model + CSV and run the audit."""
    import pickle
    from pathlib import Path

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required for the CLI. pip install pandas", file=sys.stderr)
        return 1

    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is required.", file=sys.stderr)
        return 1

    # ── Load model ──────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        return 1

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Load data ────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        print(f"ERROR: target column '{args.target}' not in CSV.", file=sys.stderr)
        return 1

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # ── Run audit ────────────────────────────────────────────────────
    import ai_critic
    from ai_critic.reporters.console import print_report

    sensitive = args.sensitive.split(",") if args.sensitive else None
    weights_dict = json.loads(args.weights) if args.weights else None

    report = ai_critic.audit(
        model, X, y,
        weights=weights_dict,
        parallel=args.parallel,
        sensitive_features=sensitive,
    )

    print_report(report)

    # ── Optional HTML report ─────────────────────────────────────────
    if args.report:
        try:
            from ai_critic.reporters.html import save_html_report
            out = save_html_report(report, args.report)
            print(f"  HTML report saved → {out}")
        except ImportError:
            print("  (install ai-critic[reports] for HTML output)")

    # ── Optional JSON dump ───────────────────────────────────────────
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))

    # ── Gate ────────────────────────────────────────────────────────
    if args.gate is not None:
        try:
            ai_critic.gate(report, min_score=args.gate)
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-critic",
        description="Comprehensive ML model evaluation — beyond accuracy.",
    )
    parser.add_argument("--version", action="version", version=f"ai-critic {_version()}")

    sub = parser.add_subparsers(dest="command")

    # ── score subcommand ─────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Audit a pickled sklearn model against a CSV dataset.")
    p_score.add_argument("model", help="Path to pickled model (.pkl)")
    p_score.add_argument("data", help="Path to CSV dataset")
    p_score.add_argument("--target", required=True, help="Name of the target column in the CSV")
    p_score.add_argument(
        "--sensitive",
        default=None,
        help="Comma-separated list of sensitive feature column names for fairness evaluation",
    )
    p_score.add_argument(
        "--weights",
        default=None,
        help='JSON string to override dimension weights, e.g. \'{"fairness": 2.0}\'',
    )
    p_score.add_argument("--parallel", action="store_true", help="Run evaluators in parallel")
    p_score.add_argument("--report", default=None, metavar="FILE.html",
                         help="Save an HTML report to this path")
    p_score.add_argument("--json", action="store_true", help="Also print full JSON to stdout")
    p_score.add_argument("--gate", type=float, default=None,
                         help="Exit with code 1 if overall_score < GATE (for CI/CD)")

    args = parser.parse_args()

    if args.command == "score":
        sys.exit(_cmd_score(args))
    else:
        parser.print_help()
        sys.exit(0)


def _version() -> str:
    try:
        from ai_critic import __version__
        return __version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()