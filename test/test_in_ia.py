import sys
import os

# Adiciona a raiz do projeto ao path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from ai_critic import AICritic
from pprint import pprint

# =========================
# 1️⃣ Load dataset
# =========================
X, y = load_breast_cancer(return_X_y=True)

# =========================
# 2️⃣ Inject artificial data leakage
# =========================
leak_feature = y.reshape(-1, 1) + np.random.normal(
    0, 0.0001, size=(len(y), 1)
)
X_leaky = np.hstack([X, leak_feature])

# =========================
# 3️⃣ Over-complex model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

# =========================
# 4️⃣ Run AI Critic
# =========================
critic = AICritic(model, X_leaky, y)

# =========================
# 5️⃣ Selectable views
# =========================
print("\n================ EXECUTIVE VIEW ================\n")
pprint(critic.evaluate(view="executive"))

print("\n================ TECHNICAL VIEW ================\n")
pprint(critic.evaluate(view="technical"))

print("\n================ DETAILS VIEW ==================\n")
pprint(critic.evaluate(view="details"))

print("\n================ MULTI VIEW ====================\n")
pprint(critic.evaluate(view=["executive", "technical"]))
