from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import numpy as np

from .validation import make_cv


def evaluate(model, X, y, plot=False):
    """
    Avalia a performance do modelo usando validação cruzada
    automaticamente adequada (StratifiedKFold ou KFold).
    """

    # =========================
    # Cross-validation adaptativa
    # =========================
    cv = make_cv(y)

    scores = cross_val_score(model, X, y, cv=cv)
    mean = float(scores.mean())
    std = float(scores.std())
    suspicious = mean > 0.995

    result = {
        "cv_mean_score": mean,
        "cv_std": std,
        "suspiciously_perfect": suspicious,
        "validation_strategy": type(cv).__name__,
        "message": (
            "Perfect CV score detected — possible data leakage."
            if suspicious
            else "CV performance within expected range."
        )
    }

    # =========================
    # Learning curve
    # =========================
    if plot:
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,  # <- MESMA estratégia de validação
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        plt.figure(figsize=(6, 4))
        plt.plot(
            train_sizes,
            np.mean(train_scores, axis=1),
            label="Treino"
        )
        plt.plot(
            train_sizes,
            np.mean(test_scores, axis=1),
            label="Validação"
        )
        plt.fill_between(
            train_sizes,
            np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
            np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
            alpha=0.2
        )
        plt.xlabel("Amostra de treino")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result
