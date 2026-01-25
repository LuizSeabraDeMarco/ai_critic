from ai_critic.evaluators import (
    robustness,
    config,
    data,
    performance
)
from ai_critic.evaluators.summary import HumanSummary
from ai_critic.sessions import CriticSessionStore
from ai_critic.evaluators.scoring import compute_scores


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

    def __init__(self, model, X, y, random_state=None, session=None):
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
        session : str or None
            Optional session name for longitudinal comparison
        """
        self.model = model
        self.X = X
        self.y = y
        self.random_state = random_state
        self.session = session
        self._store = CriticSessionStore() if session else None

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
            - list : subset of views
        plot : bool
            - True : generate plots
            - False : no plots
        """

        # =========================
        # Low-level evaluator outputs
        # =========================
        details = {}

        # -------------------------
        # Data analysis
        # -------------------------
        details["data"] = data.evaluate(
            self.X,
            self.y,
            plot=plot
        )

        # -------------------------
        # Model configuration sanity
        # -------------------------
        details["config"] = config.evaluate(
            self.model,
            n_samples=details["data"]["n_samples"],
            n_features=details["data"]["n_features"]
        )

        # -------------------------
        # Performance evaluation
        # -------------------------
        details["performance"] = performance.evaluate(
            self.model,
            self.X,
            self.y,
            plot=plot
        )

        # -------------------------
        # Robustness evaluation
        # -------------------------
        details["robustness"] = robustness.evaluate(
            self.model,
            self.X,
            self.y,
            leakage_suspected=details["data"]["data_leakage"]["suspected"],
            plot=plot
        )

        # =========================
        # Human summaries
        # =========================
        human_summary = HumanSummary().generate(details)

        payload = {
            "executive": human_summary["executive_summary"],
            "technical": human_summary["technical_summary"],
            "details": details,
            "performance": details["performance"],
        }

        # =========================
        # Session persistence (optional)
        # =========================
        if self.session:
            scores = compute_scores(payload)
            payload["scores"] = scores
            self._store.save(self.session, payload)

        # =========================
        # View selector
        # =========================
        if view == "all":
            return payload

        if isinstance(view, list):
            return {k: payload[k] for k in view if k in payload}

        return payload.get(view)

    def compare_with(self, previous_session: str) -> dict:
        """
        Compare current session with a previous one.
        """

        if not self.session:
            raise ValueError("Current session name not set.")

        current = self._store.load(self.session)
        previous = self._store.load(previous_session)

        if not previous:
            raise FileNotFoundError(
                f"Session '{previous_session}' not found."
            )

        diff = {
            "global_score": {
                "current": current["scores"]["global"],
                "previous": previous["scores"]["global"],
                "delta": current["scores"]["global"] - previous["scores"]["global"],
            },
            "components": {}
        }

        for key, value in current["scores"]["components"].items():
            prev_value = previous["scores"]["components"].get(key)
            if prev_value is not None:
                diff["components"][key] = {
                    "current": value,
                    "previous": prev_value,
                    "delta": value - prev_value
                }

        return {
            "current_session": self.session,
            "previous_session": previous_session,
            "score_diff": diff,
            "note": (
                "Score deltas indicate changes in risk profile, "
                "not absolute model quality."
            )
        }

    def deploy_decision(self):
        """
        Final deployment gate.
        """

        report = self.evaluate(view="all", plot=False)

        data_risk = report["details"]["data"]["data_leakage"]["suspected"]
        perfect_cv = report["details"]["performance"]["suspiciously_perfect"]
        robustness_verdict = report["details"]["robustness"]["verdict"]
        structural_warnings = report["details"]["config"]["structural_warnings"]

        blocking_issues = []
        risk_level = "low"

        # Hard blockers
        if data_risk and perfect_cv:
            blocking_issues.append(
                "Data leakage combined with suspiciously perfect CV score"
            )
            risk_level = "high"

        if robustness_verdict == "misleading":
            blocking_issues.append(
                "Robustness results are misleading due to inflated baseline performance"
            )
            risk_level = "high"

        if data_risk:
            blocking_issues.append(
                "Suspected target leakage in feature set"
            )
            risk_level = "high"

        # Soft blockers
        if risk_level != "high":
            if robustness_verdict == "fragile":
                blocking_issues.append(
                    "Model performance degrades significantly under noise"
                )
                risk_level = "medium"

            if perfect_cv:
                blocking_issues.append(
                    "Suspiciously perfect cross-validation score"
                )
                risk_level = "medium"

            if structural_warnings:
                blocking_issues.append(
                    "Structural complexity risks detected in model configuration"
                )
                risk_level = "medium"

        deploy = len(blocking_issues) == 0

        confidence = 1.0
        confidence -= 0.35 if data_risk else 0
        confidence -= 0.25 if perfect_cv else 0
        confidence -= 0.25 if robustness_verdict in ("fragile", "misleading") else 0
        confidence -= 0.15 if structural_warnings else 0
        confidence = max(0.0, round(confidence, 2))

        return {
            "deploy": deploy,
            "risk_level": risk_level,
            "blocking_issues": blocking_issues,
            "confidence": confidence
        }
