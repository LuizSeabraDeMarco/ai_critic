class HumanSummary:
    """
    Builds a hierarchical, human-centered interpretation
    of the AI Critic technical report.
    """

    def generate(self, report: dict) -> dict:
        leakage = report["data"]["data_leakage"]["suspected"]
        perfect_cv = report["performance"]["suspiciously_perfect"]
        robustness_verdict = report["robustness"].get("verdict")
        structural_warnings = report["config"]["structural_warnings"]

        explainability = report.get("explainability", {})
        explain_verdict = explainability.get("verdict")
        max_feature_drop = explainability.get("max_performance_drop", 0)

        # =========================
        # Executive summary
        # =========================
        if leakage and perfect_cv:
            verdict = "❌ Unreliable"
            risk_level = "high"
            deploy = False
            main_reason = "Strong evidence of data leakage inflating model performance."
        elif explain_verdict == "feature_leakage_risk":
            verdict = "❌ Unreliable"
            risk_level = "high"
            deploy = False
            main_reason = (
                "Model behavior is dominated by a single feature, "
                "suggesting shortcut learning or leakage."
            )
        elif robustness_verdict in ("fragile", "misleading") or structural_warnings:
            verdict = "⚠️ Risky"
            risk_level = "medium"
            deploy = False
            main_reason = "Structural, robustness, or dependency-related risks detected."
        else:
            verdict = "✅ Acceptable"
            risk_level = "low"
            deploy = True
            main_reason = "No critical risks detected."

        executive_summary = {
            "verdict": verdict,
            "risk_level": risk_level,
            "deploy_recommended": deploy,
            "main_reason": main_reason,
            "one_line_explanation": (
                "Although validation accuracy is extremely high, multiple signals "
                "indicate that the model does not generalize reliably."
                if verdict == "❌ Unreliable"
                else
                "The model shows acceptable behavior under current evaluation heuristics."
            )
        }

        # =========================
        # Technical summary
        # =========================
        key_risks = []
        recommendations = []

        if leakage:
            key_risks.append(
                "Data leakage suspected due to near-perfect feature–target correlation."
            )
            recommendations.append(
                "Audit and remove features highly correlated with the target."
            )

        if perfect_cv:
            key_risks.append(
                "Perfect cross-validation score detected (statistically unlikely)."
            )
            recommendations.append(
                "Re-run validation after leakage mitigation."
            )

        for w in structural_warnings:
            key_risks.append(w["message"])
            recommendations.append(
                "Reduce model complexity or adjust hyperparameters."
            )

        if explain_verdict == "feature_leakage_risk":
            key_risks.append(
                f"Single feature causes a {max_feature_drop:.2f} performance drop when permuted."
            )
            recommendations.append(
                "Remove or heavily regularize the dominant feature and retrain."
            )
        elif explain_verdict == "feature_dependency":
            key_risks.append(
                "Model relies disproportionately on a small subset of features."
            )
            recommendations.append(
                "Increase regularization or collect more diverse data."
            )

        if robustness_verdict == "misleading":
            key_risks.append(
                "Robustness metrics are misleading due to inflated baseline performance."
            )
            recommendations.append(
                "Fix baseline performance issues before trusting robustness metrics."
            )
        elif robustness_verdict == "fragile":
            key_risks.append(
                "Model is fragile under noise perturbations."
            )
            recommendations.append(
                "Consider regularization or simpler model architecture."
            )

        technical_summary = {
            "key_risks": key_risks or ["No significant risks detected."],
            "model_health": {
                "data_leakage": leakage,
                "suspicious_cv": perfect_cv,
                "structural_risk": bool(structural_warnings),
                "robustness_verdict": robustness_verdict,
                "explainability_verdict": explain_verdict
            },
            "recommendations": recommendations
        }

        return {
            "executive_summary": executive_summary,
            "technical_summary": technical_summary
        }
