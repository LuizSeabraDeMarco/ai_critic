from ai_critic.core.graph import EvaluationGraph
from ai_critic.core.aggregator import ScoreAggregator
from ai_critic.plugins.registry import EvaluatorRegistry
from ai_critic.visualization.graphviz_renderer import render_graph


class AICritic:

    def __init__(self):

        # Load all registered evaluator plugins
        evaluators = list(EvaluatorRegistry.get_all())

        # Build evaluation graph dynamically
        self.graph = EvaluationGraph(evaluators)

        self.aggregator = ScoreAggregator()

    def evaluate(self, model, X, y):
        """
        Run full AI evaluation pipeline.
        """

        dataset = {
            "X": X,
            "y": y
        }

        context = {
            "model": model,
            "dataset": dataset
        }

        results = {}

        # Execute evaluators
        for evaluator in self.graph.nodes:

            result = evaluator.evaluate(model, dataset, context)

            results[evaluator.name] = result

        scores = self.aggregator.aggregate(results)

        return {
            "scores": scores,
            "details": results
        }

    def visualize(self, output_path="evaluation_graph", format="png"):
        """
        Generate a visual representation of the evaluation graph.
        """

        return render_graph(self.graph, output_path=output_path, format=format)