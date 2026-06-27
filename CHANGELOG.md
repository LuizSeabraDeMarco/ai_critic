# Changelog

All notable changes to `ai-critic` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [4.0.0] — 2024-XX-XX

### Added
- **pandas DataFrame support** — `audit(model, df, df["target"])` now works natively;
  column names propagate to all evaluators and appear in the fairness report.
- **`sensitive_features` argument** — `audit(model, X, y, sensitive_features=["gender", "age_group"])`
  pins specific columns for the fairness evaluator instead of auto-detection.
- **HTML reporter** — `from ai_critic.reporters import save_html_report; save_html_report(report, "report.html")`
  generates a self-contained dark-theme HTML report.
- **CLI** — `ai-critic score model.pkl data.csv --target y --report out.html --gate 0.75`
  (was declared in pyproject.toml but never implemented).
- **Feature names in explainability** — when a DataFrame is passed, the permutation
  importance report now shows column names instead of `feature_0, feature_1, …`.

### Changed
- `__version__` unified to `2.0.0` (was `4.0.0` in pyproject.toml, `1.0.0` in `__init__.py`).
- `FairnessEvaluator` group column detection now also accepts float columns whose
  values are integer-like (e.g. `0.0 / 1.0` encoded as float).
- `AuditPipeline` passes `feature_names` through context to all evaluators.
- `reporters.__init__` now exports both `print_report` and `save_html_report`.

### Fixed
- Fairness evaluator group keys are now strings, not ints — fixes JSON serialisation.
- `parallel=True` now correctly propagates `feature_names` context.

---

## [1.0.0] — initial release
- 7-dimension audit: performance, robustness, explainability, calibration,
  data_quality, fairness, complexity.
- `gate()` helper for CI/CD.
- Console reporter.