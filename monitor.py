# monitor.py
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/query_logs.json")

class QueryMonitor:
    def __init__(self):
        self.query_count = 0
        self.total_latency = 0.0
        LOG_FILE.parent.mkdir(exist_ok=True)

    def log_query(self, question: str, answer: str, latency: float, sources: int):
        self.query_count += 1
        self.total_latency += latency
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "latency": latency,
            "sources": sources,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_event(self, event_type: str, data: dict):
        # Optional: log PDF uploads etc.
        pass

    def get_query_count(self) -> int:
        return self.query_count

    def get_avg_latency(self) -> float:
        if self.query_count == 0:
            return 0.0
        return round(self.total_latency / self.query_count, 2)

    def read_logs(self) -> str:
        if not LOG_FILE.exists():
            return "No logs yet."
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        logs = []
        for line in lines[-20:]:
            try:
                entry = json.loads(line)
                logs.append(f"{entry['timestamp']} | Q: {entry['question'][:60]} | {entry['latency']}s")
            except:
                continue
        return "\n".join(logs) if logs else "No readable log entries."