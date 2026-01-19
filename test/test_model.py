from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from ai_critic import AICritic


def test_ai_critic_runs():
    X, y = load_iris(return_X_y=True)

    model = LogisticRegression(max_iter=200)

    critic = AICritic(model, X, y)
    report = critic.evaluate()

    assert "performance" in report
    assert "robustness" in report
    assert report["performance"]["cv_mean_score"] > 0.5
