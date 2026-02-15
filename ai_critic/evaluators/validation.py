import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def infer_problem_type(y):
    y = np.asarray(y)
    unique_values = np.unique(y)

    if np.issubdtype(y.dtype, np.integer) or len(unique_values) <= 20:
        return "classification"

    return "regression"


def make_cv(y, n_splits=3, random_state=42):
    problem_type = infer_problem_type(y)

    if problem_type == "classification":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
