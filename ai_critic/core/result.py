from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NodeResult:
    score: float
    details: Dict[str, Any]
