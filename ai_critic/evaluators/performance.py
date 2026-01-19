from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, X, y, plot=False):
    # CV básico
    scores = cross_val_score(model, X, y, cv=3)
    mean = float(scores.mean())
    std = float(scores.std())
    suspicious = mean > 0.995

    result = {
        "cv_mean_score": mean,
        "cv_std": std,
        "suspiciously_perfect": suspicious,
        "message": (
            "Perfect CV score detected — possible data leakage."
            if suspicious else "CV performance within expected range."
        )
    }

    # =========================
    # Learning curve
    # =========================
    if plot:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)
        )
        plt.figure(figsize=(6,4))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Treino")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validação")
        plt.fill_between(train_sizes,
                         np.mean(test_scores, axis=1)-np.std(test_scores, axis=1),
                         np.mean(test_scores, axis=1)+np.std(test_scores, axis=1),
                         alpha=0.2)
        plt.xlabel("Amostra de treino")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result
