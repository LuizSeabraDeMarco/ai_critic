from typing import Dict, Any, List, Optional
from ai_critic.core.graph import EvaluationGraph
from ai_critic.core.aggregator import ScoreAggregator
from ai_critic.core.result import EvaluationReport
from ai_critic.plugins.registry import EvaluatorRegistry
from ai_critic.visualization.graphviz_renderer import render_graph

# Import default evaluators to trigger registration
import ai_critic.evaluators.performance
import ai_critic.evaluators.robustness
import ai_critic.evaluators.explainability


class AICritic:
    """
    Main entry point for the AI Critic library.
    Coordinates evaluation graph, aggregation, and reporting.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the critic with optional weights for evaluators.
        """
        # Load all registered evaluator plugins
        evaluators = list(EvaluatorRegistry.get_all())

        # Build evaluation graph dynamically
        self.graph = EvaluationGraph(evaluators)

        # Initialize aggregator with weights
        self.aggregator = ScoreAggregator(weights=weights)

    def evaluate(self, model: Any, X: Any, y: Any, parallel: bool = False) -> EvaluationReport:
        """
        Run full AI evaluation pipeline on a model and dataset.
        
        Args:
            model: The ML model to evaluate.
            X: Features dataset.
            y: Target labels.
            parallel: Whether to run independent evaluators in parallel.
            
        Returns:
            An EvaluationReport containing scores and detailed diagnostics.
        """
        dataset = {
            "X": X,
            "y": y
        }

        # Execute evaluators in topological order (optionally in parallel)
        results = self.graph.run(model, dataset, parallel=parallel)

        # Aggregate results into a final score
        scores = self.aggregator.aggregate(results, self.graph.nodes)

        # Create structured report
        return EvaluationReport(
            scores=scores,
            details=results,
            metadata={
                "nodes_executed": list(results.keys()),
                "total_nodes": len(self.graph.nodes),
                "execution_mode": "parallel" if parallel else "sequential"
            }
        )

    def visualize(self, output_path: str = "evaluation_graph", format: str = "png") -> str:
        """
        Generate a visual representation of the evaluation graph.
        """
        return render_graph(self.graph, output_path=output_path, format=format)
