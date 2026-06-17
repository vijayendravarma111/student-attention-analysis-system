# Computer Vision-Based Student Attention Analysis System - Technical Documentation

This document describes the design specifications, pipeline mechanics, and engineering metrics for the **Computer Vision-Based Student Attention Analysis System**.

---

## 🏗️ 1. Architecture Overview
The system is constructed as a modular **two-stage computer vision pipeline** that integrates spatial detection, geometry-based pose solvers, and deep learning classification:

1. **Input Interface:** Supports static image files, video files, and real-time webcam streams.
2. **Stage 1 (Face Detection & Landmarks):** Employs YuNet in ONNX format to detect faces and extract 5 key coordinates (eyes, nose, mouth corners).
3. **Head Pose Estimator:** Solves Perspective-n-Point (PnP) using standard 3D anthropometric facial nodes to calculate pitch, yaw, and roll.
4. **Spatial Preprocessing:** Crops face regions, performs bilinear resizing to $224 \times 224$, and normalizes channels.
5. **Stage 2 (Classification):** Passes the preprocessed crop to a TensorFlow/Keras neural network.
6. **Attention Scoring Engine:** Integrates classification probability with head pose deviation to compute a combined score.
7. **Analytics Dashboard:** Streams annotated video and runs real-time statistical aggregation.

---

## 📐 2. Head Pose Estimation Mechanics
Using YuNet's 5 facial landmarks, the system maps 2D coordinates to a generic 3D face model:

$$\mathbf{P}_{3D} = \begin{bmatrix} 
\text{Right Eye} \\ \text{Left Eye} \\ \text{Nose Tip} \\ \text{Right Mouth} \\ \text{Left Mouth} 
\end{bmatrix} = \begin{bmatrix} 
-35.0 & 35.0 & -50.0 \\ 
35.0 & 35.0 & -50.0 \\ 
0.0 & 0.0 & 0.0 \\ 
-25.0 & -35.0 & -40.0 \\ 
25.0 & -35.0 & -40.0 
\end{bmatrix} \text{ (in mm)}$$

### Camera Matrix Model
Focal length ($f$) is approximated as the image width ($W$).
$$\mathbf{K} = \begin{bmatrix} 
W & 0 & W/2 \\ 
0 & W & H/2 \\ 
0 & 0 & 1 
\end{bmatrix}$$

### Perspective-n-Point Solver
Using `cv2.solvePnP`, we compute the rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$.
We apply the Rodrigues transform to obtain the rotation matrix $\mathbf{R}$:
$$\mathbf{R} = \text{Rodrigues}(\mathbf{r})$$

### Euler Angles Extraction
From $\mathbf{R}$, Euler angles are decomposed as follows:
$$\theta_{\text{pitch}} = \text{atan2}(R_{2,1}, R_{2,2})$$
$$\theta_{\text{yaw}} = \text{atan2}(-R_{2,0}, \sqrt{R_{0,0}^2 + R_{1,0}^2})$$
$$\theta_{\text{roll}} = \text{atan2}(R_{1,0}, R_{0,0})$$

Thresholds ($|\theta| > 15^\circ$) determine if the user is looking Left, Right, Up, or Down.

---

## 📊 3. Attention Score Composition
The combined **Attentiveness Score ($A$)** is calculated from the CNN prediction score ($\hat{y} \in [0.0, 1.0]$, representing distraction probability) and the head pose penalties:

1. **Base Attention Score:**
   $$A_{\text{base}} = (1 - \hat{y}) \times 100$$
2. **Pose Penalties:**
   $$P_{\text{yaw}} = \max\left(0, \frac{|\theta_{\text{yaw}}| - 15^\circ}{30^\circ}\right)$$
   $$P_{\text{pitch}} = \max\left(0, \frac{|\theta_{\text{pitch}}| - 15^\circ}{30^\circ}\right)$$
   $$P_{\text{total}} = \min(0.8, P_{\text{yaw}} + P_{\text{pitch}})$$
3. **Combined Attentiveness Score:**
   $$A = A_{\text{base}} \times (1 - P_{\text{total}})$$

Students with $A \ge 50.0\%$ are classified as **Focused**; otherwise, they are marked **Not Focused**.

---

## 🛠️ 4. Codebase Architecture
The implementation is structured into modular components:

* `detector.py`: Wraps YuNet face detection and landmark localization.
* `pose.py`: Executes the PnP solver and Euler decomposition.
* `classifier.py`: Evaluates focus predictions using custom SavedModel and outlines optional MobileNetV2 alternative paths.
* `analytics.py`: Logs session statistics (Average score, focused/distracted percentage, time).
* `pipeline.py`: Orchestrates frame processing.
* `app.py`: Sets up Flask routes, video streaming threads, and the visual HTML dashboard.

---

## 📈 5. Model Evaluation Methodology
To assess the model's reliability, a validation framework is defined utilizing key classification metrics:

1. **Accuracy:** Evaluates the overall correctness of focus and distraction labeling across the dataset.
2. **Precision:** Measures the model's accuracy when predicting that a student is "Focused" or "Not Focused".
3. **Recall (Sensitivity):** Assesses the model's capability to detect all distracted/unfocused instances.
4. **F1-Score:** The harmonic mean of precision and recall, serving as a single optimization objective.
