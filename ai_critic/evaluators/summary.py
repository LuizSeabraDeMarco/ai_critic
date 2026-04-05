class HumanSummary:

    def generate(self, report: dict) -> dict:

        details = report.get("details", {})

        data = details.get("data", {})
        perf = details.get("performance", {})
        robustness = details.get("robustness", {})
        config = details.get("config", {})
        explain = details.get("explainability", {})

        leakage = data.get("data_leakage", {}).get("suspected", False)
        perfect_cv = perf.get("suspiciously_perfect", False)

        if leakage or perfect_cv:
            verdict = "❌ Unreliable"
            deploy = False
        elif robustness.get("verdict") == "fragile":
            verdict = "⚠️ Risky"
            deploy = False
        else:
            verdict = "✅ Acceptable"
            deploy = True

        return {
            "executive_summary": {
                "verdict": verdict,
                "deploy_recommended": deploy
            }
        }