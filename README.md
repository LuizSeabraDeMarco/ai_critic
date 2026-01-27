# ai-critic 🧠: The Quality Gate for Machine Learning Models

**ai-critic** is a specialized **decision-making** tool designed to audit the reliability and readiness for deployment of **scikit-learn**, **PyTorch**, and **TensorFlow** models.

Instead of merely measuring performance (accuracy, F1 score), **ai-critic** acts as a **Quality Gate**, actively probing the model to uncover *hidden risks* that commonly cause production failures — such as **data leakage**, **structural overfitting**, and **fragility under noise**.

> **ai-critic does not ask “How good is this model?”**
> It asks **“Can this model be trusted?”**

---

## 🚀 Getting Started (The Basics)

This section is ideal for beginners who need a **fast and reliable verdict** on a trained model.

### Installation

Install directly from PyPI:

```bash
pip install ai-critic
```

---

### The Quick Verdict

With just a few lines of code, you obtain an **executive-level assessment** and a **deployment recommendation**.

```python
from ai_critic import AICritic
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 1. Prepare data and model
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(max_depth=5, random_state=42)

# 2. Initialize the Critic
critic = AICritic(model, X, y)

# 3. Run the audit (executive mode)
report = critic.evaluate(view="executive")

print(f"Verdict: {report['verdict']}")
print(f"Risk Level: {report['risk_level']}")
print(f"Main Reason: {report['main_reason']}")
```

**Expected Output (example):**

```text
Verdict: ⚠️ Risky
Risk Level: medium
Main Reason: Structural or robustness-related risks detected.
```

This output is intentionally **conservative**.
If **ai-critic** recommends deployment, it means meaningful risks were *not* detected.

---

## 💡 Understanding the Critique (The Intermediary)

For data scientists who want to understand **why** the model received a given verdict and **how to improve it**.

---

### The Four Pillars of the Audit

**ai-critic** evaluates models across four independent risk dimensions:

| Pillar                 | Main Risk Detected                     | Internal Module          |
| ---------------------- | -------------------------------------- | ------------------------ |
| 📊 **Data Integrity**  | Target Leakage & Correlation Artifacts | `evaluators.data`        |
| 🧠 **Model Structure** | Over-complexity & Misconfiguration     | `evaluators.config`      |
| 📈 **Performance**     | Suspicious CV or Learning Curves       | `evaluators.performance` |
| 🧪 **Robustness**      | Sensitivity to Noise                   | `evaluators.robustness`  |

Each pillar contributes signals used later in the **deployment gate**.

---

### Full Technical & Visual Analysis

To access **all internal diagnostics**, including plots and recommendations, use `view="all"`.

```python
full_report = critic.evaluate(view="all", plot=True)

technical_summary = full_report["technical"]

print("\n--- Key Risks Detected ---")
for i, risk in enumerate(technical_summary["key_risks"], start=1):
    print(f"{i}. {risk}")

print("\n--- Recommendations ---")
for rec in technical_summary["recommendations"]:
    print(f"- {rec}")
```

Generated plots may include:

* Feature correlation heatmaps
* Learning curves
* Robustness degradation charts

---

### Robustness Test (Noise Injection)

A model that collapses under small perturbations is **not production-safe**.

```python
robustness = full_report["details"]["robustness"]

print("\n--- Robustness Analysis ---")
print(f"Original CV Score: {robustness['cv_score_original']:.4f}")
print(f"Noisy CV Score: {robustness['cv_score_noisy']:.4f}")
print(f"Performance Drop: {robustness['performance_drop']:.4f}")
print(f"Verdict: {robustness['verdict']}")
```

**Possible Verdicts:**

* `stable` → acceptable degradation
* `fragile` → high sensitivity to noise
* `misleading` → performance likely inflated by leakage

---

## ⚙️ Integration and Governance (The Advanced)

This section targets **MLOps engineers**, **architects**, and teams operating automated pipelines.

---

### Multi-Framework Support

**ai-critic 1.0+** supports models from multiple frameworks with the **same API**:

```python
# PyTorch Example
import torch
import torch.nn as nn
from ai_critic import AICritic

X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

critic = AICritic(model, X, y, framework="torch", adapter_kwargs={"epochs":5, "batch_size":64})
report = critic.evaluate(view="executive")
print(report)

# TensorFlow Example
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(20,)),
    tf.keras.layers.Dense(2)
])
critic = AICritic(model, X.numpy(), y.numpy(), framework="tensorflow", adapter_kwargs={"epochs":5})
report = critic.evaluate(view="executive")
print(report)
```

> No need to rewrite evaluation code — **one Critic API works for sklearn, PyTorch, or TensorFlow**.

---

### The Deployment Gate (`deploy_decision`)

The `deploy_decision()` method aggregates *all detected risks* and produces a final gate decision.

```python
decision = critic.deploy_decision()

if decision["deploy"]:
    print("✅ Deployment Approved")
else:
    print("❌ Deployment Blocked")

print(f"Risk Level: {decision['risk_level']}")
print(f"Confidence Score: {decision['confidence']:.2f}")

print("\nBlocking Issues:")
for issue in decision["blocking_issues"]:
    print(f"- {issue}")
```

**Conceptual model:**

* **Hard Blockers** → deployment denied
* **Soft Blockers** → deployment discouraged
* **Confidence Score (0–1)** → heuristic trust level

---

### Modes & Views (API Design)

The `evaluate()` method supports **multiple modes** via the `view` parameter:

| View          | Description                        |
| ------------- | ---------------------------------- |
| `"executive"` | High-level verdict (non-technical) |
| `"technical"` | Risks & recommendations            |
| `"details"`   | Raw evaluator outputs              |
| `"all"`       | Complete payload                   |

Example:

```python
critic.evaluate(view="technical")
critic.evaluate(view=["executive", "performance"])
```

---

### Session Tracking & Model Comparison

You can persist evaluations and compare model versions over time.

```python
critic_v1 = AICritic(model, X, y, session="v1")
critic_v1.evaluate()

critic_v2 = AICritic(model, X, y, session="v2")
critic_v2.evaluate()

comparison = critic_v2.compare_with("v1")
print(comparison["score_diff"])
```

This enables:

* Regression tracking
* Risk drift detection
* Governance & audit trails

---

### Best Practices & Use Cases

| Scenario                | Recommended Usage                      |
| ----------------------- | -------------------------------------- |
| **CI/CD**               | Block merges using `deploy_decision()` |
| **Model Tuning**        | Use technical view for guidance        |
| **Governance**          | Persist session outputs                |
| **Stakeholder Reports** | Share executive summaries              |

---

## 🔒 API Stability

Starting from version **1.0.0**, the public API of **ai-critic** follows semantic versioning.
Breaking changes will only occur in major releases.

---

## 📄 License

Distributed under the **MIT License**.

---

## 🧠 Final Note

> **ai-critic is not a benchmarking tool.**
> It is a *decision-making system*.

A failed audit does **not** mean the model is bad — it means the model **is not ready to be trusted**.

The purpose of **ai-critic** is to introduce *structured skepticism* into machine learning workflows — exactly where it belongs.
