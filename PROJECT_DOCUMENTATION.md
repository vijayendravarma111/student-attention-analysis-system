# Computer Vision-Based Student Attention Analysis System - Technical Documentation

### 🌐 Live Web Demo: [student-attention-analysis-system.onrender.com](https://student-attention-analysis-system.onrender.com)

---

## 🏗️ 1. System Overview
The **Computer Vision-Based Student Attention Analysis System** is a real-time, non-intrusive monitoring tool designed to analyze student engagement in remote learning environments. The application is structured as a modular, multi-stage computer vision pipeline:

1. **Stage 1 (Face Detection & Landmarks):** Employs YuNet in ONNX format to localize faces and extract 5 landmark coordinates (eyes, nose, mouth corners) at high frame rates.
2. **Head Pose Estimator (solvePnP):** Computes 3D head rotation angles (pitch, yaw, roll) using standard reference coordinates.
3. **Preprocessing Engine:** Scales localized facial crops to $224 \times 224$ and normalizes pixels.
4. **Stage 2 (Classification - ONNX Runtime):** Runs a lightweight classification model to determine focus state.
5. **Attention Score Engine:** Merges focus probability and head turn angles to output a combined score (0-100%).
6. **Analytics Dashboard:** Visualizes processed video feeds (static uploads, webcam, files) and logs session statistics.

---

## 🧠 2. Deep Learning Conversion (TensorFlow to ONNX Runtime)
### The Problem: OOM (Out Of Memory) Crashes
Cloud container hosting platforms (such as Render Free Tier) impose strict memory allocations of **512 MB RAM**. Running the monolithic TensorFlow/Keras framework requires importing massive runtime engines, resulting in a memory footprint of **~750 MB**, which triggers immediate OOM kills (producing a `502 Bad Gateway` error).

### The Solution: ONNX Compilation
We compiled the Keras `focus_saved_model` into the Open Neural Network Exchange (ONNX) format:

$$\text{SavedModel} \xrightarrow{\text{tf2onnx}} \text{focus_model.onnx}$$

By replacing `tensorflow` and `keras` with `onnxruntime`, the system:
* Reduced memory consumption from **~750MB to <120MB** (an 85% saving).
* Decreased container build image size from **~3.5GB to ~400MB**.
* Reduced model load times from **~10 seconds to ~0.5 seconds**.

---

## 📐 3. Head Pose Estimation & Coordinate Alignment
The system maps 2D landmark coordinates to a standard 3D anthropometric facial model.

### 3D Coordinate Mapping
To prevent sign errors and mathematical inversion (such as a $180^\circ$ pitch tilt offset), coordinates are aligned with the image spatial orientation where the Y-axis points downwards:

$$\mathbf{P}_{3D} = \begin{bmatrix} 
\text{Right Eye (viewer's left)} \\ \text{Left Eye (viewer's right)} \\ \text{Nose Tip} \\ \text{Right Mouth Corner} \\ \text{Left Mouth Corner} 
\end{bmatrix} = \begin{bmatrix} 
-35.0 & -35.0 & -50.0 \\ 
35.0 & -35.0 & -50.0 \\ 
0.0 & 0.0 & 0.0 \\ 
-25.0 & 35.0 & -40.0 \\ 
25.0 & 35.0 & -40.0 
\end{bmatrix} \text{ (in mm)}$$

### Perspective-n-Point Solver
Using `cv2.solvePnP` with the `cv2.SOLVEPNP_EPNP` flag (optimized for 5 points), we calculate the rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$. We decompose the rotation matrix $\mathbf{R} = \text{Rodrigues}(\mathbf{r})$ into Euler angles:
$$\theta_{\text{pitch}} = \text{atan2}(R_{2,1}, R_{2,2})$$
$$\theta_{\text{yaw}} = \text{atan2}(-R_{2,0}, \sqrt{R_{0,0}^2 + R_{1,0}^2})$$
$$\theta_{\text{roll}} = \text{atan2}(R_{1,0}, R_{0,0})$$

Thresholds ($|\theta| > 15^\circ$) determine if the student is looking Left, Right, Up, or Down.

---

## 📊 4. Attentiveness Score Formula
The **Attentiveness Score ($A$)** merges CNN prediction values ($\hat{y} \in [0, 1]$, representing distraction probability) with head pose penalties:

1. **Base Attention Score:**
   $$A_{\text{base}} = (1 - \hat{y}) \times 100$$
2. **Pose Penalties:**
   $$P_{\text{yaw}} = \max\left(0, \frac{|\theta_{\text{yaw}}| - 15^\circ}{30^\circ}\right)$$
   $$P_{\text{pitch}} = \max\left(0, \frac{|\theta_{\text{pitch}}| - 15^\circ}{30^\circ}\right)$$
   $$P_{\text{total}} = \min(0.8, P_{\text{yaw}} + P_{\text{pitch}})$$
3. **Combined Attentiveness Score:**
   $$A = A_{\text{base}} \times (1 - P_{\text{total}})$$

---

## ⚡ 5. Client-Side Webcam Stream Optimization
To ensure responsiveness when deployed on remote servers:
* **Sequential Ping-Pong Requests:** Instead of an interval timer firing requests blindly, the browser only captures and uploads the next frame after receiving the response from the previous frame. This completely eliminates queuing delays and lag.
* **Payload Downscaling:** Frames are resized in the browser to $480 \times 360$ and compressed using JPEG compression at 50% quality. This reduces the upload payload from **~150KB to ~15KB per frame**, speeding up transmission by **10x**.
