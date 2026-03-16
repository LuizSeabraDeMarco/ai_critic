from graphviz import Digraph

def render_graph(graph):

    dot = Digraph()

    for node in graph.nodes:
        dot.node(node.name)

    for edge in graph.edges:
        dot.edge(edge.source, edge.target)

    dot.render("evaluation_graph", format="png")