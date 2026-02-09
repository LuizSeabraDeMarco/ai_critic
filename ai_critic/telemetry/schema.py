from dataclasses import dataclass
from typing import Dict, Literal

Verdict = Literal["OK", "WARNING", "BLOCK"]

@dataclass
class CriticSignals:
    leakage: bool
    perfect_cv: bool
    robustness: str
    structural_risk: str


@dataclass
class CriticScorecard:
    critic_version: str
    model_type: str
    framework: str

    n_samples: int
    n_features: int

    score: int
    verdict: Verdict
    risk_level: Literal["low", "medium", "high"]
    block_deploy: bool

    signals: CriticSignals
    recommended_actions: list[str]
