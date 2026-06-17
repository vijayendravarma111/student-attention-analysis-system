import time

class AttentionAnalytics:
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets session tracking variables.
        """
        self.start_time = time.time()
        self.total_frames = 0
        self.total_faces_detected = 0
        self.attention_scores = []
        self.focused_count = 0
        self.distracted_count = 0

    def update(self, faces_info):
        """
        Updates analytics values with metrics from the current frame.
        """
        self.total_frames += 1
        num_faces = len(faces_info)
        self.total_faces_detected += num_faces
        
        for face in faces_info:
            score = face["attention_score"]
            self.attention_scores.append(score)
            if face["label"] == "Focused":
                self.focused_count += 1
            else:
                self.distracted_count += 1

    def get_summary(self):
        """
        Calculates and returns session statistics.
        """
        duration = time.time() - self.start_time
        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0.0
        total_predictions = self.focused_count + self.distracted_count
        focused_pct = (self.focused_count / total_predictions * 100.0) if total_predictions > 0 else 0.0
        distracted_pct = (self.distracted_count / total_predictions * 100.0) if total_predictions > 0 else 0.0

        return {
            "session_duration_sec": round(duration, 2),
            "total_frames_processed": self.total_frames,
            "average_attention_score": round(avg_score, 2),
            "focused_percentage": round(focused_pct, 2),
            "distracted_percentage": round(distracted_pct, 2),
            "total_faces_analyzed": total_predictions
        }
