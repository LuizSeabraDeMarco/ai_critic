from ai_critic import AICritic
from ai_critic.audit import audit
from ai_critic.ai_suggestions.rules import SuggestionEngine
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# 1. Load data
data = load_iris()
X, y = data.data, data.target

# 2. Train a simple model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 3. Initialize AICritic with custom weights
# Let's give more weight to robustness
weights = {
    "performance": 1.0,
    "robustness": 2.0,
    "explainability": 0.5
}
critic = AICritic(weights=weights)

# 4. Run evaluation with parallel execution
print("\n--- Running Evaluation (Parallel) ---")
report = critic.evaluate(model, X, y, parallel=True)

# 5. Show structured report summary
print(report.summary())

# 6. Show dynamic suggestions
print("\n--- AI Suggestions ---")
suggestions = SuggestionEngine.suggest(report)
for s in suggestions:
    print(f"[{s['priority']}] {s['category'].upper()}: {s['message']}")

# 7. Run full audit
print("\n--- Running Full Audit ---")
audit_result = audit(model, X, y)
print(f"Audit Status: {audit_result['report']['scores']['verdict']}")
print(f"Dataset Size Check: {audit_result['audit_checks']['dataset']['size']['message']}")
print(f"Overfitting Check: {audit_result['audit_checks']['model']['overfitting']['message']}")

print("\nEvaluation complete! The library is now on another level.")
