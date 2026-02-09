class FeedbackStore:
    def __init__(self):
        self.storage = []

    def add(self, session_id, report, success: bool):
        self.storage.append({
            "session": session_id,
            "success": success,
            "report": report
        })

    def stats(self):
        positives = sum(1 for x in self.storage if x["success"])
        negatives = sum(1 for x in self.storage if not x["success"])

        return {
            "total": len(self.storage),
            "positives": positives,
            "negatives": negatives
        }

    def all(self):
        return self.storage
