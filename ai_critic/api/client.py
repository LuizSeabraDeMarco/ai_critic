from engine.runner import CriticRunner


class AICritic:
    def __init__(self):
        self.runner = CriticRunner()

    def evaluate(self, input_data, model_output):
        return self.runner.run(input_data, model_output)