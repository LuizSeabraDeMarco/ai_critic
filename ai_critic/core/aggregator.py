class ScoreAggregator:

    def aggregate(self, results: dict) -> dict:
        scores = []

        for node in results.values():
            if "score" in node:
                scores.append(node["score"])

        if not scores:
            return {
                "overall": 0.0,
                "verdict": "insufficient_data"
            }

        overall = sum(scores) / len(scores)

        if overall > 0.9:
            verdict = "excellent"
        elif overall > 0.75:
            verdict = "good"
        elif overall > 0.6:
            verdict = "moderate"
        else:
            verdict = "poor"

        return {
            "overall": float(overall),
            "verdict": verdict
        }
