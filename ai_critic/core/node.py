from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class EvaluationNode(ABC):
    """
    Base class for all evaluation nodes in the graph.
    Unifies the concept of a node and a plugin.
    """
    name: str = "base_node"
    dependencies: List[str] = []
    weight: float = 1.0  # Used for score aggregation

    def __init__(self):
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def evaluate(self, model: Any, dataset: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the evaluation logic.
        
        Args:
            model: The ML model to evaluate.
            dataset: Dictionary containing 'X' and 'y'.
            context: Optional context containing results from dependency nodes.
            
        Returns:
            Dict containing 'score' (0-1) and other diagnostic metadata.
        """
        pass

    def __repr__(self):
        return f"<EvaluationNode: {self.name} (weight={self.weight})>"
