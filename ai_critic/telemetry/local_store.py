import json
from pathlib import Path

STORE = Path.home() / ".ai_critic" / "telemetry.jsonl"
STORE.parent.mkdir(exist_ok=True)

def save(event: dict):
    with open(STORE, "a") as f:
        f.write(json.dumps(event) + "\n")
