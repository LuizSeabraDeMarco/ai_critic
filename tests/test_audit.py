"""
Basic smoke tests — run with:  python -m pytest tests/ -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import model_audit
from model_audit.core.types import Verdict


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def iris_model():
    data = load_iris()
    m = RandomForestClassifier(n_estimators=20, random_state=0).fit(data.data, data.target)
    return m, data.data, data.target


@pytest.fixture
def diabetes_model():
    data = load_diabetes()
    m = LinearRegression().fit(data.data, data.target)
    return m, data.data, data.target


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def test_audit_returns_report(iris_model):
    model, X, y = iris_model
    report = model_audit.audit(model, X, y)
    assert report.overall_score >= 0.0
    assert report.overall_score <= 1.0
    assert report.overall_verdict in list(Verdict)


def test_all_dimensions_present(iris_model):
    model, X, y = iris_model
    report = model_audit.audit(model, X, y)
    expected = {"performance", "robustness", "explainability",
                "data_quality", "calibration", "complexity", "fairness"}
    assert expected.issubset(set(report.dimensions.keys()))


def test_regression_audit(diabetes_model):
    model, X, y = diabetes_model
    report = model_audit.audit(model, X, y)
    assert "performance" in report.dimensions
    perf = report.dimensions["performance"]
    # R² > 0 for this dataset
    assert perf.score > 0.0


def test_to_dict(iris_model):
    model, X, y = iris_model
    report = model_audit.audit(model, X, y)
    d = report.to_dict()
    assert "overall_score" in d
    assert "dimensions" in d
    assert "problem_type" in d


def test_gate_passes(iris_model):
    model, X, y = iris_model
    report = model_audit.audit(model, X, y)
    # Should not raise for a decent RF on Iris
    model_audit.gate(report, min_score=0.0)


def test_gate_blocks():
    X, y = make_classification(n_samples=200, n_features=20, random_state=0)
    # Dummy model — always predicts 0
    from sklearn.dummy import DummyClassifier
    m = DummyClassifier(strategy="most_frequent").fit(X, y)
    report = model_audit.audit(m, X, y)
    with pytest.raises(RuntimeError):
        model_audit.gate(report, min_score=0.99)


def test_parallel_matches_sequential(iris_model):
    model, X, y = iris_model
    r_seq = model_audit.audit(model, X, y, parallel=False)
    r_par = model_audit.audit(model, X, y, parallel=True)
    # Scores should be very close (not identical due to randomness in robustness)
    assert abs(r_seq.overall_score - r_par.overall_score) < 0.15
