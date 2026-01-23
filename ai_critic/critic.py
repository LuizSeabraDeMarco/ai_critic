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

    Produces a multi-layered risk assessment including:
    - Data integrity analysis
    - Model configuration sanity checks
    - Performance evaluation (CV + learning curves)
    - Robustness & leakage heuristics
    - Human-readable executive and technical summaries
    """

    def __init__(self, model, X, y, random_state=None):
        """
        Parameters
        ----------
        model : sklearn-compatible estimator
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        random_state : int or None
            Global seed for reproducibility (optional)
        """
        self.model = model
        self.X = X
        self.y = y
        self.random_state = random_state

    def evaluate(self, view="all", plot=False):
        """
        Evaluate the model.

        Parameters
        ----------
        view : str or list
            - "all" : full payload (default)
            - "executive" : executive summary only
            - "technical" : technical summary only
            - "details" : low-level evaluator outputs
            - list : subset of views (e.g. ["executive", "details"])
        plot : bool
            - True : generate plots (learning curve, heatmap, robustness)
            - False : no plots

        Returns
        -------
        dict
            Evaluation payload according to selected view
        """

        # =========================
        # Low-level evaluator outputs
        # =========================
        details = {}

        # -------------------------
        # Data analysis
        # -------------------------
        data_report = data.evaluate(
            self.X,
            self.y,
            plot=plot
        )
        details["data"] = data_report

        # -------------------------
        # Model configuration sanity
        # -------------------------
        details["config"] = config.evaluate(
            self.model,
            n_samples=data_report["n_samples"],
            n_features=data_report["n_features"]
        )

        # -------------------------
        # Performance evaluation
        # (CV strategy inferred automatically)
        # -------------------------
        details["performance"] = performance.evaluate(
            self.model,
            self.X,
            self.y,
            plot=plot
        )

        # -------------------------
        # Robustness & leakage analysis
        # -------------------------
        details["robustness"] = robustness.evaluate(
            self.model,
            self.X,
            self.y,
            leakage_suspected=data_report["data_leakage"]["suspected"],
            plot=plot
        )

        # =========================
        # Human-centered summaries
        # =========================
        human_summary = HumanSummary().generate(details)

        # =========================
        # Full payload (PUBLIC API)
        # =========================
        payload = {
            "executive": human_summary["executive_summary"],
            "technical": human_summary["technical_summary"],
            "details": details,
            # Convenience shortcut (prevents KeyError in user code)
            "performance": details["performance"]
        }

        # =========================
        # View selector
        # =========================
        if view == "all":
            return payload

        if isinstance(view, list):
            return {k: payload[k] for k in view if k in payload}

        return payload.get(view)
