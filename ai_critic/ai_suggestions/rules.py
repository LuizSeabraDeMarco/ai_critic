class SuggestionEngine:

    @staticmethod
    def suggest(report):

        suggestions = []

        details = report.get("details", {})
        perf = details.get("performance", {})
        robustness = details.get("robustness", {})
        explain = details.get("explainability", {})

        if perf.get("suspiciously_perfect"):
            suggestions.append("Check for data leakage")

        if robustness.get("verdict") == "fragile":
            suggestions.append("Improve robustness with regularization")

        if explain.get("verdict") == "feature_dependency":
            suggestions.append("Reduce feature dependency")

        if not suggestions:
            suggestions.append("Model looks good")

        return suggestions