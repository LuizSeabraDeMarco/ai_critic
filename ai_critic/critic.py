from typing import Dict, Any, Optional

from ai_critic.core.graph import EvaluationGraph
from ai_critic.core.aggregator import ScoreAggregator
from ai_critic.core.schema import build_report

from ai_critic.plugins.registry import EvaluatorRegistry

# evaluators (auto-register)
import ai_critic.evaluators.performance
import ai_critic.evaluators.robustness
import ai_critic.evaluators.explainability
import ai_critic.evaluators.data
import ai_critic.evaluators.config

# post-processing
from ai_critic.evaluators.scoring import compute_scores
from ai_critic.evaluators.summary import HumanSummary
from ai_critic.ai_suggestions.rules import SuggestionEngine


class AICritic:
    """
    Unified AI evaluation engine.
    Produces a single standardized report.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        evaluators = list(EvaluatorRegistry.get_all())
        self.graph = EvaluationGraph(evaluators)
        self.aggregator = ScoreAggregator(weights=weights)

    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        parallel: bool = False
    ) -> Dict[str, Any]:

        dataset = {"X": X, "y": y}

        # 1. Run evaluation graph
        raw_results = self.graph.run(model, dataset, parallel=parallel)

        # 2. Aggregate base scores (0–1)
        scores = self.aggregator.aggregate(raw_results, self.graph.nodes)

        # 3. Build unified report schema
        report = build_report(raw_results, scores)

        # 4. Risk scoring (0–100 layer)
        report["risk"] = compute_scores(report)

        # 5. Human-readable summary
        report["summary"] = HumanSummary().generate(report)

        # 6. Suggestions / recommendations
        report["suggestions"] = SuggestionEngine.suggest(report)

        return report