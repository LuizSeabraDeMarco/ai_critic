# evaluators/adapters.py
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

class ModelAdapter:
    """
    Wraps scikit-learn, PyTorch, or TensorFlow models to provide a
    unified fit/predict interface for AICritic.
    """

    def __init__(self, model, framework="sklearn", **kwargs):
        """
        Parameters
        ----------
        model : object
            The original model (sklearn estimator, torch.nn.Module, or tf.keras.Model)
        framework : str
            One of "sklearn", "torch", "tensorflow"
        kwargs : dict
            Extra hyperparameters for training (epochs, batch_size, optimizer, etc)
        """
        self.model = model
        self.framework = framework.lower()
        self.kwargs = kwargs

        if self.framework not in ("sklearn", "torch", "tensorflow"):
            raise ValueError(f"Unsupported framework: {framework}")

        # PyTorch default settings
        if self.framework == "torch":
            self.epochs = kwargs.get("epochs", 5)
            self.lr = kwargs.get("lr", 1e-3)
            self.loss_fn = kwargs.get("loss_fn", nn.MSELoss())
            self.optimizer_class = kwargs.get("optimizer", torch.optim.Adam)
            self.device = kwargs.get("device", "cpu")
            self.model.to(self.device)

        # TensorFlow default settings
        if self.framework == "tensorflow":
            self.epochs = kwargs.get("epochs", 5)
            self.batch_size = kwargs.get("batch_size", 32)
            self.loss_fn = kwargs.get("loss_fn", "mse")
            self.optimizer = kwargs.get("optimizer", "adam")
            self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

    def fit(self, X, y):
        if self.framework == "sklearn":
            self.model.fit(X, y)
        elif self.framework == "torch":
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device).view(-1, 1)
            optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)

            self.model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                output = self.model(X_tensor)
                loss = self.loss_fn(output, y_tensor)
                loss.backward()
                optimizer.step()
        elif self.framework == "tensorflow":
            self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        if self.framework == "sklearn":
            return self.model.predict(X)
        elif self.framework == "torch":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                return self.model(X_tensor).cpu().numpy().flatten()
        elif self.framework == "tensorflow":
            return self.model.predict(X).flatten()
