import cv2
import numpy as np
from detector import FaceDetector
from pose import PoseEstimator
from classifier import FocusClassifier
from analytics import AttentionAnalytics

class AttentionAnalysisPipeline:
    def __init__(self, detector_path="face_detection_yunet_2023mar.onnx", classifier_path="focus_model.onnx", architecture="custom"):
        """
        Orchestrator for the two-stage computer vision pipeline.
        Combines detection, pose estimation, classification, and metrics extraction.
        """
        self.detector = FaceDetector(detector_path)
        self.pose_estimator = PoseEstimator()
        self.classifier = FocusClassifier(classifier_path, architecture)
        self.analytics = AttentionAnalytics()

    def process_frame(self, frame_bgr):
        """
        Processes a single input frame.
        Applies face detection, landmarks pose estimation, classification, and metrics overlays.
        Returns:
            - annotated_frame_bgr: frame with visual annotations (bounding boxes, landmarks, scores)
            - faces_info: list of dictionaries detailing the state of each detected face
        """
        h, w, _ = frame_bgr.shape
        faces = self.detector.detect(frame_bgr)
        faces_info = []

        # Convert frame to RGB for classification and internal model calculations
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_frame = frame_rgb.copy()

        if faces is not None:
            for face in faces:
                x, y, bw, bh = map(int, face[:4])
                
                # Boundary constraints mapping
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                if bw <= 0 or bh <= 0:
                    continue

                face_crop = frame_rgb[y:y+bh, x:x+bw]
                if face_crop.size == 0:
                    continue

                # Stage 2: Deep Learning Classifier Inference
                pred = self.classifier.predict(face_crop)

                # Stage 1: Landmarks extraction and head pose estimation
                pitch, yaw, roll, pose_direction = self.pose_estimator.estimate_pose(face, w, h)

                # Compute combined Attention Score (0-100)
                base_score = (1.0 - pred) * 100.0
                
                # Apply pose penalty: penalize score if head is looking away (> 15 degrees)
                yaw_penalty = max(0.0, (abs(yaw) - 15.0) / 30.0)
                pitch_penalty = max(0.0, (abs(pitch) - 15.0) / 30.0)
                total_penalty = min(0.8, yaw_penalty + pitch_penalty) # Max penalty capped at 80%
                
                attention_score = base_score * (1.0 - total_penalty)
                attention_score = max(0.0, min(100.0, attention_score))

                # Binary classification threshold
                if attention_score >= 50.0:
                    label = "Focused"
                    color = (0, 255, 0) # Green (RGB)
                else:
                    label = "Not Focused"
                    color = (255, 0, 0) # Red (RGB)

                faces_info.append({
                    "bbox": (x, y, bw, bh),
                    "pred_raw": float(pred),
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll),
                    "pose_direction": pose_direction,
                    "attention_score": float(attention_score),
                    "label": label
                })

                # Visual Rendering: Draw face bounding box
                cv2.rectangle(annotated_frame, (x, y), (x+bw, y+bh), color, 2)
                
                # Draw facial landmark points
                for idx in range(5):
                    lx = int(face[4 + idx*2])
                    ly = int(face[5 + idx*2])
                    cv2.circle(annotated_frame, (lx, ly), 3, (0, 255, 255), -1)

                # Construct visual text labels
                label_text = f"{label} ({attention_score:.1f}%)"
                pose_text = f"P:{pitch:.1f} Y:{yaw:.1f} ({pose_direction})"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Dynamic text placement
                cv2.putText(annotated_frame, label_text, (x, y - 25 if y > 35 else y + bh + 15), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(annotated_frame, pose_text, (x, y - 10 if y > 20 else y + bh + 30), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Update Session Analytics
        self.analytics.update(faces_info)

        # Convert annotated frame back to BGR for OpenCV display/saving/streaming compatibility
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        return annotated_frame_bgr, faces_info
