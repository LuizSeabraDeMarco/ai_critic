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
    Produces a multi-layered risk assessment with visualizations.
    """

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def evaluate(self, view="all", plot=False):
        """
        Evaluate the model.

        Parameters:
        -----------
        view : str or list
            - "all" : return full payload
            - "executive" : only executive summary
            - "technical" : only technical summary
            - "details" : only low-level module outputs
            - list : subset of views
        plot : bool
            - True : generate plots (learning curve, heatmap, robustness)
            - False : no plots
        """

        # =========================
        # Low-level technical details
        # =========================
        details = {}

        # Data analysis + heatmap
        data_report = data.evaluate(self.X, self.y, plot=plot)
        details["data"] = data_report

        # Model configuration
        details["config"] = config.evaluate(
            self.model,
            n_samples=data_report["n_samples"],
            n_features=data_report["n_features"]
        )

        # Performance + learning curve
        details["performance"] = performance.evaluate(
            self.model, self.X, self.y, plot=plot
        )

        # Robustness + CV clean vs noisy
        details["robustness"] = robustness.evaluate(
            self.model,
            self.X,
            self.y,
            leakage_suspected=data_report["data_leakage"]["suspected"],
            plot=plot
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
