# ai-critic 3.0.0

`pip install ai-critic`

Latest version
Released: 2026

AI Critic — Evaluation Graph Engine for ML models.

---

## Navigation

* Project description
* Release history
* Download files

---

## Verified details

Maintainer
Luiz Filipe Seabra de Marco

---

## Unverified details

**License:** MIT License (MIT)
**Author:** Luiz Filipe Seabra de Marco
**Tags:** machine learning, model evaluation, ml validation, robustness, explainability, cross validation, ai audit, ml scoring, evaluation engine
**Requires:** Python >=3.8
**Provides-Extra:** dev

### Classifiers

Development Status
5 - Production/Stable

Intended Audience
Developers
Science/Research

License
OSI Approved :: MIT License

Operating System
OS Independent

Programming Language
Python :: 3
Python :: 3.8
Python :: 3.9
Python :: 3.10
Python :: 3.11

Topic
Software Development :: Libraries
Scientific/Engineering :: Artificial Intelligence

---

# Project description

# AI Critic: The Evaluation Graph Engine for Machine Learning

AI Critic is a modular, graph-based evaluation engine designed to analyze machine learning models in a structured, extensible, and deterministic way.

Instead of providing isolated metrics, AI Critic executes an **Evaluation Graph** composed of independent evaluation nodes. Each node analyzes one dimension of model quality — such as performance, robustness, or explainability — and produces standardized outputs.

The final result is an aggregated score with a clear verdict.

In summary:

You provide a model, data (X, y), and AI Critic executes a structured evaluation pipeline that produces:

* Cross-validation diagnostics
* Robustness under noise
* Feature sensitivity analysis
* Overall quality score
* Clear deployment verdict

No telemetry.
No black-box ML meta-model.
No overengineering.

Just deterministic evaluation architecture.

---

# 🧠 Evaluation Graph Architecture

AI Critic 3.0 introduces the Evaluation Graph Engine.

Each evaluator is a node:

* PerformanceEvaluator
* RobustnessEvaluator
* ExplainabilityEvaluator

Nodes:

* Are independent
* Can declare dependencies
* Produce standardized output
* Return a normalized score

The graph executes them sequentially and aggregates results.

This architecture enables:

* Future plugin system
* Custom evaluation nodes
* Parallel execution
* Enterprise-level extensibility

---

# 🚀 Key Features

### 📊 Cross-Validation Intelligence

Automatically detects classification vs regression and selects the correct CV strategy.

* StratifiedKFold for classification
* KFold for regression
* Detects suspiciously perfect scores
* Reports validation strategy used

---

### 🛡 Robustness Under Noise

Tests model stability by injecting controlled Gaussian noise.

* Measures performance degradation
* Classifies model as stable or fragile
* Converts robustness drop into normalized score

---

### 🔍 Feature Sensitivity (Explainability Proxy)

Model-agnostic permutation analysis:

* Measures performance drop per feature
* Detects shortcut learning
* Flags potential leakage risk
* Produces explainability score

---

### 🎯 Unified Scoring System

All evaluators produce:

```
{
  "score": float,
  "verdict": str,
  ...
}
```

The ScoreAggregator computes:

* Overall score (0–1)
* Final verdict:

  * excellent
  * good
  * moderate
  * poor

---

### 🧩 Modular Graph Engine

The core engine allows:

* Adding custom evaluation nodes
* Replacing scoring strategies
* Integrating into CI pipelines
* Embedding inside ML platforms

---

# 💡 Quick Start

## Basic Usage

```python
from ai_critic import AICritic
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

critic = AICritic()
report = critic.evaluate(model, X, y)

print(report["scores"])
```

Output:

```
{
  "overall": 0.87,
  "verdict": "good"
}
```

---

# 📈 Detailed Output Structure

```python
{
  "scores": {
      "overall": 0.83,
      "verdict": "good"
  },
  "details": {
      "performance": {...},
      "robustness": {...},
      "explainability": {...}
  }
}
```

Each evaluator returns structured diagnostic metadata.

---

# 🖥 CLI Usage

```
ai-critic --model model.pkl --data dataset.csv --target label
```

Output:

```
=== AI CRITIC REPORT ===

Overall score: 0.812
Verdict: good
```

JSON mode:

```
ai-critic --model model.pkl --data dataset.csv --target label --json
```

---

# 🧪 Evaluation Dimensions

## 1️⃣ Performance

* Cross-validation mean score
* Standard deviation
* Suspiciously perfect detection

## 2️⃣ Robustness

* Noise injection test
* Performance drop calculation
* Stability classification

## 3️⃣ Explainability

* Feature permutation sensitivity
* Shortcut detection
* Leakage risk signal

---

# ⚙️ Installation

```
pip install ai-critic
```

Dependencies:

* scikit-learn
* numpy
* matplotlib (optional for visualization)

---

# 🏗 Extending AI Critic

You can create custom nodes:

```python
from ai_critic.core.node import EvaluationNode

class FairnessEvaluator(EvaluationNode):

    name = "fairness"
    dependencies = []

    def evaluate(self, context):
        return {
            "score": 0.9,
            "verdict": "acceptable"
        }
```

Then inject into the graph:

```python
critic.graph = EvaluationGraph([
    PerformanceEvaluator(),
    RobustnessEvaluator(),
    ExplainabilityEvaluator(),
    FairnessEvaluator()
])
```

---

# 🎯 Design Philosophy

AI Critic is built on three principles:

1. Deterministic evaluation
2. Structural modularity
3. No hidden learning layer

It is not an AutoML system.
It is not a model trainer.

It is an evaluation engine.

---

# 📄 License

Distributed under the MIT License. See LICENSE for more information.