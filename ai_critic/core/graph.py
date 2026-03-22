import concurrent.futures
from typing import Dict, Any, List, Optional, Set
from collections import deque
from .node import EvaluationNode


class EvaluationGraph:
    """
    Manages the execution of evaluation nodes based on their dependencies.
    Supports both sequential and parallel execution.
    """

    def __init__(self, nodes: List[EvaluationNode]):
        self.nodes = {node.name: node for node in nodes}
        self.execution_order = self._resolve_execution_order()

    def run(self, model: Any, dataset: Dict[str, Any], parallel: bool = False) -> Dict[str, Any]:
        """
        Run all nodes in the correct order.
        """
        if parallel:
            return self._run_parallel(model, dataset)
        return self._run_sequential(model, dataset)

    def _run_sequential(self, model: Any, dataset: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for name in self.execution_order:
            node = self.nodes[name]
            node_context = {dep: results[dep] for dep in node.dependencies if dep in results}
            results[name] = node.evaluate(model, dataset, context=node_context)
        return results

    def _run_parallel(self, model: Any, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute independent nodes in parallel using a thread pool.
        """
        results = {}
        completed: Set[str] = set()
        
        # Track in-degrees for dynamic scheduling
        in_degree = {name: len([d for d in node.dependencies if d in self.nodes]) 
                     for name, node in self.nodes.items()}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while len(completed) < len(self.nodes):
                # Find nodes that are ready to run
                ready_to_run = [name for name, degree in in_degree.items() 
                                if degree == 0 and name not in completed]
                
                if not ready_to_run:
                    # This should not happen if graph is a DAG
                    break
                
                # Submit ready nodes to executor
                future_to_name = {}
                for name in ready_to_run:
                    node = self.nodes[name]
                    node_context = {dep: results[dep] for dep in node.dependencies if dep in results}
                    future = executor.submit(node.evaluate, model, dataset, node_context)
                    future_to_name[future] = name
                    # Mark as "running" by setting in_degree to -1
                    in_degree[name] = -1

                # Wait for at least one to complete
                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    results[name] = future.result()
                    completed.add(name)
                    
                    # Update in-degrees of dependents
                    for other_name, other_node in self.nodes.items():
                        if name in other_node.dependencies:
                            in_degree[other_name] -= 1

        return results

    def _resolve_execution_order(self) -> List[str]:
        adj = {name: [] for name in self.nodes}
        in_degree = {name: 0 for name in self.nodes}

        for name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in self.nodes:
                    adj[dep].append(name)
                    in_degree[name] += 1

        queue = deque([name for name in self.nodes if in_degree[name] == 0])
        order = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("Circular dependency detected in evaluation graph.")
        return order

    def __repr__(self):
        return f"<EvaluationGraph: {len(self.nodes)} nodes>"
