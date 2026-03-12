import json
import os
import time


class JsonLogger:
    """Append-only JSONL logger for experiment tracking."""

    def __init__(self, exp_dir: str, filename: str = "logs.jsonl"):
        self.path = os.path.join(exp_dir, filename)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self, data: dict):
        """Append a JSON record with timestamp."""
        data["_timestamp"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def read_all(self) -> list:
        """Read all logged records."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
