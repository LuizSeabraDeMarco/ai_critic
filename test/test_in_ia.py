from ai_critic import AICritic
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier(max_depth=12, random_state=42)

critic = AICritic(model, X, y)
report = critic.evaluate(plot=True)
print(report["executive"])
print(report["technical"])
