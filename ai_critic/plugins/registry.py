class EvaluatorRegistry:

    _evaluators = {}

    @classmethod
    def register(cls, evaluator):
        cls._evaluators[evaluator.name] = evaluator

    @classmethod
    def get_all(cls):
        return cls._evaluators.values()