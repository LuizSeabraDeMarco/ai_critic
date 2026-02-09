# ai_critic/critic.py

from ai_critic.evaluators import (
    robustness,
    config,
    data,
    performance,
    adapters
)
from ai_critic.evaluators.summary import HumanSummary
from ai_critic.sessions import CriticSessionStore
from ai_critic.evaluators.scoring import compute_scores

from ai_critic.learning import (
    extract_features,
    CriticModel,
    CriticTrainer,
    policy_decision,
    recommend_changes
)
from ai_critic.feedback import FeedbackStore


class AICritic:
    def __init__(
        self,
        model,
        X,
        y,
        random_state=None,
        session=None,
        framework="sklearn",
        adapter_kwargs=None
    ):
        adapter_kwargs = adapter_kwargs or {}
        self.framework = framework.lower()

        self.model = (
            adapters.ModelAdapter(model, framework=self.framework, **adapter_kwargs)
            if self.framework != "sklearn"
            else model
        )

        self.X = X
        self.y = y
        self.session = session

        self.ml_model = CriticModel()
        try:
            self.ml_model.load()
        except Exception:
            pass

        self.trainer = CriticTrainer(self.ml_model)
        self.feedback = FeedbackStore()
        self._store = CriticSessionStore() if session else None

    def evaluate(self, view="all", plot=False):
        details = {}

        details["data"] = data.evaluate(self.X, self.y, plot=plot)
        details["config"] = config.evaluate(
            self.model,
            n_samples=details["data"]["n_samples"],
            n_features=details["data"]["n_features"]
        )
        details["performance"] = performance.evaluate(
            self.model, self.X, self.y, plot=plot
        )
        details["robustness"] = robustness.evaluate(
            self.model,
            self.X,
            self.y,
            leakage_suspected=details["data"]["data_leakage"]["suspected"],
            plot=plot
        )

        human = HumanSummary().generate(details)

        payload = {
            "executive": human["executive_summary"],
            "technical": human["technical_summary"],
            "details": details,
            "meta": {
                "framework": self.framework,
                "n_samples": details["data"]["n_samples"],
                "n_features": details["data"]["n_features"],
            }
        }

        payload["scores"] = compute_scores(payload)

        if self.session:
            self._store.save(self.session, payload)

        return payload if view == "all" else payload.get(view)

    def deploy_decision(self, success_feedback=None):
        report = self.evaluate(view="all", plot=False)

        rule_decision = self._rule_based_decision(report)
        features = extract_features(report)
        ml_score = self.ml_model.predict_proba(features)

        decision = policy_decision(rule_decision, ml_score)
        recommendations = recommend_changes(report)

        # 🔁 FEEDBACK LOOP AUTOMÁTICO
        if success_feedback is not None:
            self.feedback.add(self.session, report, success_feedback)
            self.trainer.add_feedback(report, success_feedback)

        return {
            "deploy": decision["deploy"],
            "risk_level": rule_decision["risk_level"],
            "ml_score": round(ml_score, 3),
            "recommendations": recommendations,
            "feedback_stats": self.feedback.stats()
        }

    def _rule_based_decision(self, report):
        blocking = []
        risk = "low"

        if report["details"]["data"]["data_leakage"]["suspected"]:
            blocking.append("Data leakage suspected")
            risk = "high"

        if report["details"]["performance"]["suspiciously_perfect"]:
            blocking.append("Suspiciously perfect CV score")
            risk = "medium"

        if report["details"]["config"]["risk_level"] == "high":
            blocking.append("High structural complexity")
            risk = "medium"

        return {
            "deploy": len(blocking) == 0,
            "risk_level": risk,
            "blocking_issues": blocking
        }
