# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from spektral.layers import GlobalSumPool
import logging

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Define the CustomGCNConv layer (copied from training code)
class CustomGCNConv(tf.keras.layers.Layer):
    def __init__(self, channels, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[0][-1], self.channels),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        x, a = inputs
        eye = tf.eye(tf.shape(a)[-1], batch_shape=[tf.shape(a)[0]])
        a_hat = a + eye
        d = tf.reduce_sum(a_hat, axis=-1)
        d_inv_sqrt = tf.math.pow(d, -0.5)
        d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)
        d_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
        a_norm = tf.matmul(tf.matmul(d_inv_sqrt, a_hat), d_inv_sqrt)

        x = tf.matmul(a_norm, x)
        x = tf.matmul(x, self.kernel)
        x = tf.nn.bias_add(x, self.bias)
        return self.activation(x) if self.activation else x

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

# Load the pre-trained GNN model
model = load_model(
    'gnn_model.h5',
    custom_objects={
        'CustomGCNConv': CustomGCNConv,
        'GlobalSumPool': GlobalSumPool
    },
    compile=False  # Avoid metrics compilation warning
)
print("Model loaded successfully.")

# Define class names
# Update DATASET_PATH to your local dataset directory
DATASET_PATH = "G:\Sign_language_GNN\sample"  # Adjust to your actual dataset path
if os.path.exists(DATASET_PATH):
    class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
else:
    print(f"Warning: Dataset path {DATASET_PATH} not found. Using default class names.")
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]  # Default: 0-9, A-Z

print(f"Class names: {class_names}")

# Verify class names match model output
# Assuming model outputs a probability distribution over len(class_names) classes
expected_num_classes = model.output_shape[-1]
if len(class_names) != expected_num_classes:
    print(f"Warning: Number of class names ({len(class_names)}) does not match model output ({expected_num_classes}).")
    print("Predictions may be incorrect. Please verify class_names.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to create hand adjacency matrix (copied from training code)
def create_hand_adjacency():
    connections = [
        (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]
    adj = np.eye(21)
    for i, j in connections:
        adj[i,j] = 1
        adj[j,i] = 1
    return adj

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Add padding around the hand
            padding = 50
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Extract landmarks for GNN input
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            adjacency = create_hand_adjacency()

            # Prepare GNN input
            x_input = np.expand_dims(landmarks, axis=0).astype(np.float32)  # Shape: (1, 21, 3)
            a_input = np.expand_dims(adjacency, axis=0).astype(np.float32)  # Shape: (1, 21, 21)

            # Make the prediction
            try:
                prediction = model.predict([x_input, a_input], verbose=0)
                pred_index = np.argmax(prediction)
                if pred_index < len(class_names):
                    predicted_class = class_names[pred_index]
                    confidence = np.max(prediction)
                else:
                    predicted_class = "Unknown"
                    confidence = 0.0
                    print(f"Warning: Prediction index {pred_index} out of range for class_names.")

                # Draw the bounding box and prediction on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Predicted: {predicted_class} ({confidence:.2f})",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Prediction error: {str(e)}")

    # Show the frame with predictions
    cv2.imshow('Webcam Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()