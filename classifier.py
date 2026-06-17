import os
import numpy as np
import onnxruntime as ort

class FocusClassifier:
    def __init__(self, model_path="focus_model.onnx", architecture="custom"):
        """
        Initializes the focus classifier using ONNX Runtime.
        This provides a lightweight, production-grade inference engine with <10% of TensorFlow's RAM footprint.
        """
        self.architecture = architecture
        
        if not os.path.exists(model_path) and os.path.exists("focus_model.onnx"):
            model_path = "focus_model.onnx"
            
        print(f"[OK] Loading classifier: {model_path} via ONNX Runtime.")
        
        # Initialize ONNX inference session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.is_loaded = True

    def predict(self, face_rgb):
        """
        Preprocesses face crop and runs ONNX inference.
        """
        import cv2

        # Preprocessing: Spatial resizing & pixel normalization
        face_img = cv2.resize(face_rgb, (224, 224))
        face_img = face_img.astype(np.float32) / 255.0
        face_input = np.expand_dims(face_img, axis=0) # Shape: (1, 224, 224, 3)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: face_input})
        pred = outputs[0][0][0]

        return float(pred)
