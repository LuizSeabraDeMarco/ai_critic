import json
from pathlib import Path
from datetime import datetime


class CriticSessionStore:
    """
    Simple local persistence layer for ai-critic sessions.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(
            base_dir or Path.home() / ".ai_critic_sessions"
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.json"

    def save(self, name: str, payload: dict):
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload
        }
        with open(self._session_path(name), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, name: str) -> dict | None:
        path = self._session_path(name)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)["payload"]
