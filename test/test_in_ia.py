from ai_critic import AICritic
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dataset propositalmente desbalanceado
X = np.random.rand(200, 10)
y = np.array([0] * 180 + [1] * 20)

model = RandomForestClassifier(
    max_depth=12,
    random_state=42
)

critic = AICritic(model, X, y)
report = critic.evaluate(plot=False)

print("\n=== EXECUTIVE SUMMARY ===")
print(report["executive"])

print("\n=== TECHNICAL SUMMARY ===")
print(report["technical"])

print("\n=== PERFORMANCE DETAILS ===")
print(report["performance"])
