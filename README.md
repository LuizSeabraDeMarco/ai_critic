Performance under noise

> Visualizations are optional and do not affect the decision logic.

---

## ⚙️ Main API

### `AICritic(model, X, y)`

* `model`: scikit-learn compatible estimator
* `X`: feature matrix
* `y`: target vector

### `evaluate(view="all", plot=False)`

* `view`: `"executive"`, `"technical"`, `"details"`, `"all"` or custom list
* `plot`: generates graphs when `True`

---

## 🧠 What ai-critic Detects

| Category | Risks |

| ------------ | ---------------------------------------- |

| 🔍 Data | Target Leakage, NaNs, Imbalance |

| 🧱 Structure | Excessive Complexity, Overfitting |

| 📈 Validation | Perfect or Statistically Suspicious CV |

| 🧪 Robustness | Stable, Fragile, or Misleading |

---

## 🛡️ Best Practices

* **CI/CD:** Use executive output as a *quality gate*
* **Iteration:** Use technical output during tuning
* **Governance:** Log detailed output
* **Skepticism:** Never blindly trust a perfect CV

---

## 🧭 Use Cases

* Pre-deployment Audit
* ML Governance
* CI/CD Pipelines
* Risk Communication for Non-Technical Users

---

## 📄 License

Distributed under the **MIT License**.

---

## 🧠 Final Note

**ai-critic** is not a *benchmarking* tool. It's a **decision-making tool**.

If a model fails here, it doesn't mean it's bad—it means it **shouldn't be trusted yet**.