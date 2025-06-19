import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os

# Import the custom GCN layer (essential for loading the model)
# Ensure CustomGCNConv class is defined exactly as it was during training.
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from spektral.layers import GlobalSumPool


# Re-define CustomGCNConv and _create_hand_adjacency exactly as in the training script
# This is crucial for Keras to correctly load custom layers.

class CustomGCNConv(tf.keras.layers.Layer):
    def __init__(self, channels, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_feature_dim, self.channels),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        x, a = inputs
        eye = tf.eye(tf.shape(a)[-1], batch_shape=[tf.shape(a)[0]], dtype=a.dtype)
        a_hat = a + eye
        d = tf.reduce_sum(a_hat, axis=-1)
        d_inv_sqrt = tf.pow(d + 1e-10, -0.5)
        d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)
        d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
        a_norm = tf.matmul(tf.matmul(d_mat_inv_sqrt, a_hat), d_mat_inv_sqrt)

        x = tf.matmul(x, self.kernel)
        output = tf.matmul(a_norm, x)
        output = output + self.bias  # Add bias here
        return self.activation(output) if self.activation else output

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "activation": tf.keras.activations.serialize(self.activation)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _create_hand_adjacency():
    """
    Defines the adjacency matrix based on standard MediaPipe hand landmark connections.
    This matrix represents the fixed graph structure of a human hand.
    Must be identical to the one used during training.
    """
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Connections across palm (metacarpals)
    ]
    adj = np.eye(21, dtype=np.float32)
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


# --- Configuration ---
# Set the path to your trained model file.
# Make sure this matches the name the training script saved the GNN model as.
# The training script saved it as "gnn_model_full_precision.h5"
MODEL_PATH_H5 = "hand_gesture_model_012_only.h5"

# Load class names from the dataset directory
# This MUST match the order of classes during training.
DATASET_PATH = "G:/Sign_language_GNN/model/imagedata-main"  # Same path as in training script


def load_class_names(dataset_path):
    """Loads class names from the dataset directory."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found. Cannot determine class names.")
        return []
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    if not class_names:
        print(f"Warning: No class subdirectories found in '{dataset_path}'. Prediction might be inaccurate.")
    return class_names


CLASS_NAMES = load_class_names(DATASET_PATH)


def load_trained_model(model_path_h5):
    """
    Loads the trained Keras model from .h5 format.
    Custom objects (CustomGCNConv, GlobalSumPool) must be provided.
    """
    custom_objects = {
        'CustomGCNConv': CustomGCNConv,
        'GlobalSumPool': GlobalSumPool  # Spektral's GlobalSumPool also needs to be passed
    }

    model = None
    if os.path.exists(model_path_h5):
        print(f"Loading model from {model_path_h5}...")
        try:
            model = tf.keras.models.load_model(model_path_h5, custom_objects=custom_objects)
            print("Model loaded successfully from .h5 format.")
        except Exception as e:
            print(f"Error loading .h5 model: {e}")
            model = None

    if model is None:
        raise FileNotFoundError(
            f"No trained model found at the specified .h5 path: {model_path_h5}. Please ensure training was successful and this file exists.")

    return model


def main():
    if not CLASS_NAMES:
        print(
            "Exiting: Class names not loaded. Please ensure DATASET_PATH is correct and contains class subdirectories.")
        return

    try:
        # Pass only the .h5 model path
        model = load_trained_model(MODEL_PATH_H5)
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # Process video stream
        max_num_hands=1,  # Detect one hand
        min_detection_confidence=0.7,  # Higher confidence for real-time
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Get the fixed adjacency matrix (same as during training)
    adjacency_matrix = _create_hand_adjacency()
    # Expand dims for batch processing (batch size 1)
    adjacency_matrix_batch = np.expand_dims(adjacency_matrix, axis=0)

    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting real-time ISL recognition. Press 'q' to quit.")
    print("Ensure your hand is clearly visible to the camera.")

    prev_frame_time = 0
    new_frame_time = 0

    # Store recent predictions for smoothing (optional)
    prediction_history = []
    history_length = 5  # Number of recent predictions to consider

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip frame horizontally for a more natural mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image_rgb)

        # Draw hand landmarks and perform inference
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Red points
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green connections
                )

                # Extract landmarks (x, y, z) and normalize them (they are already normalized by MediaPipe)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

                # Expand dims for batch processing (batch size 1)
                landmarks_batch = np.expand_dims(landmarks, axis=0)

                # Perform prediction using the loaded model
                # The GNN model expects two inputs: [node_features, adjacency_matrix]
                predictions = model.predict([landmarks_batch, adjacency_matrix_batch], verbose=0)

                # Get the predicted class index
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]

                # Add prediction to history for smoothing
                prediction_history.append(predicted_class_idx)
                if len(prediction_history) > history_length:
                    prediction_history.pop(0)

                # Simple smoothing: take the most frequent prediction in history
                if prediction_history:
                    # Using bincount for efficiency with integer arrays
                    counts = np.bincount(prediction_history)
                    smoothed_class_idx = np.argmax(counts)
                else:
                    smoothed_class_idx = predicted_class_idx  # Fallback if history is empty

                # Get the class name
                if 0 <= smoothed_class_idx < len(CLASS_NAMES):
                    predicted_sign = CLASS_NAMES[smoothed_class_idx]
                else:
                    predicted_sign = "UNKNOWN"

                # Display the prediction and confidence
                text = f"{predicted_sign} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # If no hand is detected, clear history or indicate "No Hand"
            prediction_history.clear()
            cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),
                    2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('ISL Real-time Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nReal-time recognition stopped.")


if __name__ == "__main__":
    main()