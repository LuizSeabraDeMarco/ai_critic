import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

class CriticModel:
    def __init__(self, path="critic_model.joblib"):
        self.path = path
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        joblib.dump(self.model, self.path)

    def load(self):
        self.model = joblib.load(self.path)
        self.is_trained = True

    def predict_proba(self, features: dict) -> float:
        if not self.is_trained:
            return 0.5  # neutro

        X = np.array([list(features.values())])
        return float(self.model.predict_proba(X)[0][1])
