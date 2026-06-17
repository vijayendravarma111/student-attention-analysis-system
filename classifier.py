import os
import tensorflow as tf
from keras.layers import TFSMLayer

class FocusClassifier:
    def __init__(self, model_path="focus_saved_model", architecture="custom"):
        """
        Initializes focus classifier.
        Supports:
        - "custom": Custom trained SavedModel loaded using TFSMLayer (functional default)
        - "mobilenetv2": MobileNetV2 pre-trained base + custom top layer (evaluation architecture)
        """
        self.architecture = architecture
        if architecture == "custom" and os.path.exists(model_path):
            layer = TFSMLayer(model_path, call_endpoint="serving_default")
            inp = tf.keras.Input(shape=(224, 224, 3))
            out = layer(inp)
            self.model = tf.keras.Model(inp, out)
            self.is_loaded = True
            print("[OK] Loaded custom Focus SavedModel.")
        else:
            # Fallback/Evaluation alternative: MobileNetV2 architecture
            print(f"[WARN] Loading MobileNetV2 evaluation architecture. (Note: Needs fine-tuning on attentiveness dataset)")
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights="imagenet"
            )
            base_model.trainable = False
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = tf.keras.Model(inputs, outputs)
            self.is_loaded = True

    def predict(self, face_rgb):
        """
        Preprocesses face crop and returns probability prediction.
        """
        import cv2
        import numpy as np

        # Preprocessing: Spatial scaling & Min-Max Normalization
        face_img = cv2.resize(face_rgb, (224, 224))
        face_img = face_img / 255.0
        face_input = np.expand_dims(face_img, axis=0)

        if self.architecture == "custom":
            output = self.model(face_input, training=False)
            if isinstance(output, dict):
                pred = list(output.values())[0].numpy()[0][0]
            else:
                pred = output.numpy()[0][0]
        else:
            pred = self.model(face_input, training=False).numpy()[0][0]

        return float(pred)
