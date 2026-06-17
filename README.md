#  Computer Vision-Based Student Attention Analysis System

###  Live Web Demo: [student-attention-analysis-system.onrender.com](https://student-attention-analysis-system.onrender.com)

---

##  Why this project matters 
In digital classrooms and remote learning, instructors lose the ability to read the room. They can't easily see who is focused, who is distracted, or who has drifted away. 

This project solves that problem. By combining **Deep Learning** and **3D Head Geometry**, it acts as a passive, privacy-preserving analyzer that monitors attentiveness in real-time. It doesn't record videos; it simply interprets the physical signals of attention—head orientation, gaze alignment, and facial focus markers—and provides a quantitative score of classroom engagement.

---

##  Key Project Capabilities
* **Live Webcam Monitoring:** Runs fully inside the web browser with zero-lag transmission to the backend.
* **3D Head Pose Tracking:** Resolves exactly where a student is looking (pitch, yaw, roll) to identify if they are looking away (left, right, up, down).
* **Attentiveness Scoring (0-100%):** Calculates a combined score based on facial deep learning confidence and head rotation angles.
* **Continuous Analytics:** Displays real-time session statistics (average focus score, focused vs. distracted ratio over time).

---

##  Under the Hood (How it Works)

The system works as a clean **two-stage computer vision pipeline**:

```
[Webcam/Video Ingest] 
      ↓
[Stage 1: Face Detection (YuNet ONNX)] ──→ Extracts 5 Facial Landmarks (Eyes, Nose, Mouth)
      ↓
[3D Head Pose Solver (solvePnP)] ──→ Calculates Pitch, Yaw, Roll and determines looking direction
      ↓
[Stage 2: Focus Classification (ONNX)] ──→ Evaluates facial expression focus probability
      ↓
[Attention Score Engine] ──→ Combines ML probability with head orientation penalties
      ↓
[Real-Time Analytics Dashboard]
```

1. **Face & Landmark Localization:** The fast **YuNet ONNX** model detects the face and localizes 5 landmark points (eyes, nose, mouth corners).
2. **Head Pose Estimation:** Using OpenCV's **Perspective-n-Point (PnP)** solver, the system maps the 2D facial landmarks against a standard 3D human head model to calculate head rotation angles.
3. **Attentiveness Classifier:** The cropped face image is normalized and passed to a deep learning binary classifier (compiled to ONNX format) to predict the focus state.
4. **Combined Scoring:** The system merges the deep learning prediction with head-turn penalties (e.g. looking away drops the score) to output a final 0-100% attention score.

---

##  Technical Challenges Overcome

### 1. Migrating to ONNX Runtime (Resolved 502 Bad Gateway)
* **Challenge:** Deploying a standard TensorFlow/Keras model to Render's free tier caused the service to crash immediately (OOM / 502 Bad Gateway) due to Render's strict **512MB RAM limit**.
* **Solution:** Converted the Keras SavedModel into an optimized **ONNX format (`focus_model.onnx`)** and migrated the backend code to **ONNX Runtime**. This reduced the runtime memory footprint by **over 85% (from ~750MB to <120MB)**, making the app highly stable, lightweight, and fast.

### 2. Zero-Lag Webcam Streaming (Thwarted Network Queue Congestion)
* **Challenge:** Initial webcam streaming sent frames at a fixed timer interval. On slower upload connections, requests backed up in the browser's network buffer, creating a compounding video lag.
* **Solution:** Implemented a **non-blocking sequential request model (Ping-Pong loop)** where a new frame is only captured and sent after the server responds to the previous one. Coupled with 480px downscaling and 50% JPEG compression, upload size dropped by **90% (from ~150KB to ~15KB per frame)**, eliminating all lag.

### 3. Aligned 3D Coordinate Space (Corrected 180° Pitch Offset)
* **Challenge:** The PnP solver initially calculated an inverted pitch ($\approx 180^\circ$) for forward-facing users, causing false distracted labels.
* **Solution:** Re-aligned the 3D landmark points in the constructor to match the image coordinate space (where the Y-axis points down), bringing forward-facing pitch angles to a correct $\approx 0^\circ$.

---

##  Tech Stack & Modularity
* **Deep Learning Inference:** ONNX Runtime
* **Computer Vision:** OpenCV (DNN module, FaceDetectorYN, solvePnP)
* **Web Framework:** Flask
* **Math & Image Operations:** NumPy, Pillow
* **Clean Code Structure:** decoupling concerns into `detector.py`, `pose.py`, `classifier.py`, `analytics.py`, `pipeline.py`, and `app.py`.

---



### **Computer Vision-Based Student Attention Analysis System**
* **Tech Stack:** OpenCV (DNN, FaceDetectorYN), ONNX Runtime, solvePnP, Euler Decomposition, Flask, Docker, Python.
* **Bullet Points:**
  * Designed and implemented a modular **two-stage computer vision pipeline** that integrates an ONNX-optimized **YuNet face detector** with a custom **ONNX focus classifier** to evaluate student attentiveness in real-time.
  * Engineered a **3D Head Pose Estimator** utilizing Perspective-n-Point (PnP) solvers (using `cv2.SOLVEPNP_EPNP` for 5-point alignment) and Euler angle decomposition to extract pitch, yaw, and roll metrics from facial landmarks.
  * Formulated a combined **Attentiveness Score (0-100)** algorithm that incorporates base CNN classification confidence with head pose deviation penalties, capping yaw and pitch orientation anomalies.
  * Migrated the deep learning classifier from TensorFlow to **ONNX Runtime**, reducing runtime RAM usage by **85%** (from 750MB to <120MB) to allow stable Docker deployment within low-resource cloud hosting environments.
  * Built a live **analytics dashboard** featuring continuous webcam/video frame ingestion at 30 FPS via M-JPEG streams, tracking statistical session metrics (average attention, focus ratio distributions).
