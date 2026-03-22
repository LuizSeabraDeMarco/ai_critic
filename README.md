# AI Critic 4.0.0 (Enhanced Edition)

`pip install ai-critic`

AI Critic is a **graph-based evaluation engine for machine learning models**.

Instead of isolated metrics, AI Critic runs a structured **evaluation pipeline** that analyzes multiple dimensions of model quality.

## New in Version 4.0
- **Unified Core Architecture**: Streamlined `EvaluationNode` and `EvaluatorPlugin` for easier extension.
- **Advanced Graph Engine**: Real topological sorting of dependencies.
- **Parallel Execution**: Run independent evaluators in parallel for faster results.
- **Weighted Aggregation**: Control the importance of each evaluator in the final score.
- **Structured Reporting**: New `EvaluationReport` object with JSON, Dict, and Text summary support.
- **Smart Suggestion Engine**: Prioritized actionable advice based on evaluation results.

---

## Quick Start

```python
from ai_critic import AICritic
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data and train model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier().fit(X, y)

# Initialize with custom weights
weights = {"performance": 1.0, "robustness": 1.5}
critic = AICritic(weights=weights)

# Evaluate (supports parallel execution)
report = critic.evaluate(model, X, y, parallel=True)

# Print summary
print(report.summary())

# Get AI suggestions
from ai_critic.ai_suggestions.rules import SuggestionEngine
suggestions = SuggestionEngine.suggest(report)
for s in suggestions:
    print(f"[{s['priority']}] {s['category']}: {s['message']}")
```

---

## Core Components

### 1. Evaluation Graph
The engine automatically resolves dependencies between evaluators. For example, `Robustness` depends on `Performance`.

### 2. Weighted Aggregator
Individual scores are combined using a weighted average. You can define weights during `AICritic` initialization.

### 3. Evaluator Plugins
Creating custom evaluators is simple:

```python
from ai_critic.plugins.base import EvaluatorPlugin
from ai_critic.plugins.registry import EvaluatorRegistry

class FairnessEvaluator(EvaluatorPlugin):
    name = "fairness"
    weight = 1.0
    dependencies = ["performance"]

    def evaluate(self, model, dataset, context=None):
        # Your logic here
        return {"score": 0.95, "message": "Fairness is high."}

# Register the plugin
EvaluatorRegistry.register(FairnessEvaluator())
```

---

## CLI Usage
```bash
ai-critic --model model.pkl --data dataset.csv --target label --parallel
```

---

## Design Philosophy
1. **Deterministic evaluation**
2. **Modular & Parallel architecture**
3. **Transparent & Actionable diagnostics**
