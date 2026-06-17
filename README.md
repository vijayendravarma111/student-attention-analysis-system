# Computer Vision-Based Student Attention Analysis System

An end-to-end, multi-stage computer vision and deep learning system designed for real-time localization, head pose estimation, and attentiveness classification. 

This system operates as a sequential computer vision pipeline: utilizing an ONNX-optimized **YuNet Face Detector** to localize face bounding boxes and extract 5 facial landmarks, executing **Perspective-n-Point (PnP)** algorithms to resolve 3D head pose angles (pitch, yaw, roll), and passing normalized facial crops to a custom **TensorFlow/Keras classifier** to compute a combined, pose-penalized **Attentiveness Score (0-100)**. 

The application is deployed with a live Flask-based analytics dashboard, supporting static images, uploaded video files, and real-time webcam streams.

---

## 🚀 Key Computer Vision Competencies Demonstrated
* **Multi-Stage Inference Pipelines:** Designed a pipeline sequentially performing face detection $\rightarrow$ facial landmark localization $\rightarrow$ head pose estimation $\rightarrow$ spatial crop normalization $\rightarrow$ deep learning classification $\rightarrow$ attention score composition.
* **3D Head Pose Estimation:** Implemented Perspective-n-Point (PnP) using `cv2.solvePnP` with 3D reference coordinates to resolve head orientation angles (yaw, pitch, roll) from 2D facial landmarks.
* **Continuous Frame Processing:** Built generator streams using multipart M-JPEG to feed processed video frames from webcams and files at 30 FPS.
* **Deep Learning & Model Optimization:** Configured a dual-architecture classifier framework supporting functional custom `SavedModel` nodes and MobileNetV2 feature extractors.
* **Statistical Session Analytics:** Developed an analytics module tracking average attention, focused/distracted frame distributions, and session duration.

---

## 🧠 System Pipeline Architecture

```
[Video / Webcam Input]
         ↓
[Stage 1: Face Detection (YuNet ONNX)] 
         ↓ (NMS Filtering & Bounding Box)
[Facial Landmark Localization] (5 Coordinates)
         ↓
[Head Pose Estimation (solvePnP)] ──→ [Resolves Pitch, Yaw, Roll]
         ↓                                      ↓
[Spatial Face Crop & Preprocess]         [Computes Pose Penalty]
         ↓ (Resize 224x224, Normalization)       ↓
[Stage 2: Focus Classification (TensorFlow)]    │
         ↓ (Prediction Probability)             │
[Attention Score Generation] ←──────────────────┘
         ↓ (Combined Formula)
[Analytics Dashboard & Visual Overlays]
```

---

## 🛠️ Codebase Modularity
The codebase has been refactored into a highly modular engineering structure:
* `detector.py`: Stage 1 face detection wrapping YuNet ONNX.
* `pose.py`: 3D head pose estimation using PnP and Euler decomposition.
* `classifier.py`: Stage 2 focus classification wrapping TensorFlow models.
* `analytics.py`: Track, log, and calculate session engagement statistics.
* `pipeline.py`: Orchestrates components into a unified computer vision pipeline.
* `app.py`: Serves the dashboard routes, handles file uploads, and streams webcam feeds.

---

## 📊 Evaluation & Verification Plan

Model evaluation is conducted using standard classification metrics:

| Metric | Target Performance |
|---|---|
| **Accuracy** | 91.2% |
| **Precision** | 89.5% |
| **Recall** | 92.1% |
| **F1-Score** | 90.8% |

---

## 📈 Future Enhancements
* **3D Eye Gaze Estimation:** Track iris vectors to assess if a student's gaze drifts from the primary screen space.
* **YOLOv8-Face Integration:** Upgrade detection layers to YOLOv8-face to evaluate performance scaling.
* **TensorRT/ONNX Runtime Acceleration:** Export downstream networks to TensorRT format to minimize execution latency.
