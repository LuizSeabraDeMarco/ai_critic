from ai_critic.evaluators import (
    robustness,
    config,
    data,
    performance
)
from ai_critic.evaluators.summary import HumanSummary


class AICritic:
    """
    Automated reviewer for scikit-learn models.
    Produces a multi-layered risk assessment.
    """

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def evaluate(self, view="all"):
        """
        view:
            - "all"
            - "executive"
            - "technical"
            - "details"
            - list of views
        """

        # =========================
        # Low-level technical details
        # =========================
        details = {}

        data_report = data(self.X, self.y)
        details["data"] = data_report

        details["config"] = config(
            self.model,
            n_samples=data_report["n_samples"],
            n_features=data_report["n_features"]
        )

        details["performance"] = performance(
            self.model, self.X, self.y
        )

        details["robustness"] = robustness(
            self.model,
            self.X,
            self.y,
            leakage_suspected=data_report["data_leakage"]["suspected"]
        )

        # =========================
        # Human interpretation
        # =========================
        human = HumanSummary().generate(details)

        # =========================
        # Full payload
        # =========================
        payload = {
            "executive": human["executive_summary"],
            "technical": human["technical_summary"],
            "details": details
        }

        # =========================
        # View selector
        # =========================
        if view == "all":
            return payload

        if isinstance(view, list):
            return {k: payload[k] for k in view if k in payload}

        return payload.get(view)
