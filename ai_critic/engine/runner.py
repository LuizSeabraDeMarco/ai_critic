from core.types import CriticInput
from engine.pipeline import CriticPipeline


class CriticRunner:
    def __init__(self):
        self.pipeline = CriticPipeline()

    def run(self, input_data, model_output):
        data = CriticInput(
            input_data=input_data,
            model_output=model_output
        )

        results = self.pipeline.run(data)
        score = self.pipeline.aggregate(results)

        return {
            "final_score": score,
            "details": results
        }