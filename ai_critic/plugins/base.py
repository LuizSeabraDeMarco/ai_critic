from typing import Dict, Any, Optional
from ai_critic.core.node import EvaluationNode


class EvaluatorPlugin(EvaluationNode):
    """
    Plugin-friendly wrapper for EvaluationNode.
    Inherits all core capabilities of a graph node.
    """
    
    def evaluate(self, model: Any, dataset: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plugin-specific implementation.
        Must return a dictionary with a 'score' key.
        """
        raise NotImplementedError("Plugins must implement the 'evaluate' method.")
