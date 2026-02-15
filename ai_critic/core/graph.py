from typing import Dict, Any, List
from .node import EvaluationNode


class EvaluationGraph:

    def __init__(self, nodes: List[EvaluationNode]):
        self.nodes = {node.name: node for node in nodes}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        context = {"input": input_data}
        results = {}

        for name in self._resolve_execution_order():
            node = self.nodes[name]
            output = node.evaluate(context)
            results[name] = output
            context[name] = output

        return results

    def _resolve_execution_order(self):
        # versão simples — depois pode virar topological sort real
        return list(self.nodes.keys())
