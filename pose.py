import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        # 3D model coordinates of 5 facial landmarks in millimeters (mm)
        # Reference points based on standard human head geometry:
        # 1. Right Eye (viewer's left / anatomical right)
        # 2. Left Eye (viewer's right / anatomical left)
        # 3. Nose Tip
        # 4. Right Mouth Corner (viewer's left / anatomical right)
        # 5. Left Mouth Corner (viewer's right / anatomical left)
        self.model_points = np.array([
            [-35.0, -35.0, -50.0],  # Right Eye (viewer's left / anatomical right)
            [35.0, -35.0, -50.0],   # Left Eye (viewer's right / anatomical left)
            [0.0, 0.0, 0.0],        # Nose Tip
            [-25.0, 35.0, -40.0],   # Right Mouth Corner (viewer's left / anatomical right)
            [25.0, 35.0, -40.0]     # Left Mouth Corner (viewer's right / anatomical left)
        ], dtype=np.float32)

    def estimate_pose(self, face, img_width, img_height):
        """
        Estimates the pitch, yaw, and roll angles of the head.
        landmarks coordinates extracted from YuNet output.
        """
        if len(face) < 14:
            return 0.0, 0.0, 0.0, "Forward"

        try:
            # Extract 2D coordinates of the 5 landmarks
            image_points = np.array([
                [face[4], face[5]],     # Right Eye
                [face[6], face[7]],     # Left Eye
                [face[8], face[9]],     # Nose Tip
                [face[10], face[11]],   # Right Mouth Corner
                [face[12], face[13]]    # Left Mouth Corner
            ], dtype=np.float32)

            # Approximate camera intrinsic matrix (focal length is assumed equal to image width)
            focal_length = img_width
            center = (img_width / 2.0, img_height / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            # Assuming zero lens distortion for typical webcam feeds
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            # Solve Perspective-n-Point to find camera rotation & translation vectors
            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success:
                return 0.0, 0.0, 0.0, "Forward"
        except Exception as e:
            print(f"[WARN] solvePnP failed: {e}")
            return 0.0, 0.0, 0.0, "Forward"

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles (rotation order: pitch-yaw-roll / x-y-z)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0

        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        # Determine semantic orientation based on angular thresholds
        direction = "Forward"
        if yaw > 15.0:
            direction = "Looking Left"
        elif yaw < -15.0:
            direction = "Looking Right"
        elif pitch > 15.0:
            direction = "Looking Down"
        elif pitch < -15.0:
            direction = "Looking Up"

        return pitch, yaw, roll, direction
