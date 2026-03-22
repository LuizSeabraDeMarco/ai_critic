import logging
from typing import Any, Optional

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

logger = logging.getLogger(__name__)


def render_graph(graph: Any, output_path: str = "evaluation_graph", format: str = "png") -> str:
    """
    Generate a visual representation of the evaluation graph.
    Returns the path to the generated file or an error message.
    """
    if not HAS_GRAPHVIZ:
        msg = "Graphviz is not installed. Visualization skipped."
        logger.warning(msg)
        return msg

    dot = Digraph(comment='AI Critic Evaluation Graph')
    dot.attr(rankdir='LR')  # Left to Right orientation

    # Add nodes
    for name, node in graph.nodes.items():
        label = f"{name}\n(weight={getattr(node, 'weight', 1.0)})"
        dot.node(name, label)

    # Add edges based on dependencies
    for name, node in graph.nodes.items():
        for dep in node.dependencies:
            if dep in graph.nodes:
                dot.edge(dep, name)

    try:
        output_file = dot.render(output_path, format=format, cleanup=True)
        return f"Graph rendered to {output_file}"
    except Exception as e:
        return f"Error rendering graph: {str(e)}"
