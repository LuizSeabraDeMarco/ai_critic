class EvaluatorPlugin:

    name = "base"

    def evaluate(self, model, dataset, context):
        raise NotImplementedError