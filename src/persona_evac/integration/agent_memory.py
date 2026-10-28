"""Each agent remembers only its own completed or ongoing motion outcomes."""


class AgentMemory:
    def __init__(self, agent):
        self.count = 0
        self.last = None
        self.history = []
        self.anchor = None

    @staticmethod
    def waited(r, now):
        return (
            r["agent"]["stair_wait_s"]
            + r["agent"].get("decision_wait_s", 0)
            + (max(0, now - r.get("arrived_at", now)) if r["phase"] == "queueing" else 0)
        )

    def outcome(self, r, now):
        if self.last is None:
            return None
        if r["phase"] == "evacuated":
            now = min(now, r["agent"].get("evacuation_time_s", now))
        walked = max(0, r["agent"].get("distance_walked_m", 0) - self.last["walked"])
        waited = max(0, self.waited(r, now) - self.last["waited"])
        result = (
            "evacuated"
            if r["phase"] == "evacuated"
            else "reached_next_floor"
            if r["floor"] != self.last["floor"]
            else "still_blocked"
            if waited >= 3 and walked < 0.5
            else "advanced"
            if walked >= 0.5
            else "little_progress"
        )
        return {
            "id": f"M{self.count}",
            "time_s": self.last["time"],
            "action": self.last["action"],
            "stair_id": self.last["target"],
            "floor": self.last["floor"],
            "elapsed_s": round(now - self.last["time"], 1),
            "walked_m": round(walked, 1),
            "waited_s": round(waited, 1),
            "result": result,
        }

    def memory(self, r, now):
        current = self.outcome(r, now)
        return (self.history + ([current] if current else []))[-3:]
