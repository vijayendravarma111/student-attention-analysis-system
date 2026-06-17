import cv2
import numpy as np

class FaceDetector:
    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", score_threshold=0.7, nms_threshold=0.3):
        """
        Initializes YuNet ONNX face detector.
        """
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=500
        )

    def detect(self, img_bgr):
        """
        Detects faces in a BGR image.
        Returns array of detected faces or None.
        Each face row format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rm, y_rm, x_lm, y_lm, score]
        """
        h, w, _ = img_bgr.shape
        self.detector.setInputSize((w, h))
        success, faces = self.detector.detect(img_bgr)
        return faces if success and faces is not None else None
