from ai_critic import AICritic

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# BANCO DE DADOS GENÉRICO
# =========================
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# IA SIMPLES
# =========================
model = RandomForestClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print(f"\nAccuracy do modelo: {acc:.2f}")

# =========================
# AI CRITIC
# =========================
critic = AICritic()

print("\nExecutando AI Critic...\n")

result = critic.evaluate(
    model=model,
    X=X_test,
    y=y_test
)

print("\nResultado:\n")
print(result)