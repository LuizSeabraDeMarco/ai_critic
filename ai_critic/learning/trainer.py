from .features import extract_features

class CriticTrainer:
    def __init__(self, critic_model, min_samples=10):
        self.model = critic_model
        self.min_samples = min_samples
        self.X = []
        self.y = []

    def add_feedback(self, report, success: bool):
        features = extract_features(report)
        self.X.append(list(features.values()))
        self.y.append(int(success))

        if len(self.y) >= self.min_samples:
            self.model.train(self.X, self.y)
