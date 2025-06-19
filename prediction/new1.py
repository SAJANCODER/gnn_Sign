import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from tqdm import tqdm # For progress bars (install with: pip install tqdm)
import pandas as pd # For optional CSV export (install with: pip install pandas)

# --- 1. Custom Graph Convolutional Layer Definition ---
# This class MUST be identical to the one used during your model's training
# to ensure TensorFlow can load the model correctly.
class CustomGCNConv(tf.keras.layers.Layer):
    def __init__(self, channels, activation=None, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_feature_dim, self.channels),
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
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
        x, a = inputs # x: node features, a: adjacency matrix
        eye = tf.eye(tf.shape(a)[-1], batch_shape=[tf.shape(a)[0]], dtype=a.dtype)
        a_hat = a + eye # Add self-loops to adjacency matrix
        d = tf.reduce_sum(a_hat, axis=-1) # Degree matrix
        d_inv_sqrt = tf.pow(d + 1e-10, -0.5) # Inverse square root of degrees
        d_inv_sqrt = tf.where(tf.math.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt) # Handle inf
        d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt) # Diagonal matrix of inverse square roots

        # Normalized adjacency matrix: D_hat^-0.5 * A_hat * D_hat^-0.5
        a_norm = tf.matmul(tf.matmul(d_mat_inv_sqrt, a_hat), d_mat_inv_sqrt)

        # Graph convolution operation: A_norm * X * W
        x = tf.matmul(x, self.kernel)
        output = tf.matmul(a_norm, x)
        output = output + self.bias # Add bias

        return self.activation(output) if self.activation else output

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        # This is important for custom layers when loading models
        return cls(**config)

# --- 2. Helper Functions for Prediction Process ---

def _create_hand_adjacency():
    """
    Defines the adjacency matrix based on standard MediaPipe hand landmark connections.
    This must be identical to the one used during training to ensure model compatibility.
    """
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17) # Connections across palm (metacarpals)
    ]
    num_nodes = 21 # For a single hand (MediaPipe always returns 21 landmarks for a hand)
    adj = np.eye(num_nodes, dtype=np.float32) # Add self-loops initially
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj

def extract_landmarks(image_path, mp_hands_instance, image_size=(256, 256)):
    """
    Loads an image and processes it with MediaPipe to extract hand landmarks.
    Returns landmarks (21, 3) or None if no hand is detected or image fails to load.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            # print(f"Warning: Could not read image file (might be corrupted or empty): {image_path}")
            return None

        # Convert to RGB (MediaPipe expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize for consistent MediaPipe input, matches training preprocessing
        img_resized = cv2.resize(img_rgb, image_size, interpolation=cv2.INTER_AREA)

        results = mp_hands_instance.process(img_resized)

        if results.multi_hand_landmarks:
            # Take landmarks from the first detected hand (as per training assumption)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark], dtype=np.float32)
            return landmarks
        else:
            return None # No hand detected
    except Exception as e:
        # print(f"Error processing image {image_path} for landmarks: {e}")
        return None # Error during processing

# --- 3. Main Prediction Function ---

def run_predictions_on_directory(model_path, prediction_dir_path, class_names,
                                 batch_size=32, min_detection_confidence=0.7, image_size=(256, 256)):
    """
    Loads the trained GNN model and performs batch predictions on all images
    found within the specified directory (and its subdirectories).

    Args:
        model_path (str): Path to the saved Keras model file (.h5).
        prediction_dir_path (str): Root directory containing images to predict.
        class_names (list): A list of class names in the exact order they were
                            used during model training. This is crucial for
                            correctly mapping prediction indices to labels.
        batch_size (int, optional): Number of images to process in one prediction batch.
                                    Defaults to 32.
        min_detection_confidence (float, optional): Minimum confidence for MediaPipe
                                                    to detect a hand. Defaults to 0.7.
        image_size (tuple, optional): Target size (width, height) for images before
                                      MediaPipe processing. Defaults to (256, 256).

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'image_path', 'predicted_sign', 'confidence'.
    """
    # Validate model and directory paths
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please ensure the path is correct and the model file exists.")
        return []

    if not os.path.isdir(prediction_dir_path):
        print(f"Error: Prediction directory '{prediction_dir_path}' does not exist or is not a directory.")
        return []

    # Load the trained Keras model
    try:
        # We need to provide CustomGCNConv and GlobalSumPool if they are custom layers
        # GlobalSumPool is from spektral, but sometimes needs to be explicitly mentioned.
        from spektral.layers import GlobalSumPool # Ensure spektral is installed and imported
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'CustomGCNConv': CustomGCNConv, 'GlobalSumPool': GlobalSumPool}
        )
        print(f"✅ Model loaded successfully from '{model_path}'")
    except Exception as e:
        print(f"❌ Error loading model from '{model_path}': {e}")
        print("Please ensure:")
        print("1. 'CustomGCNConv' definition in this script exactly matches the one used during training.")
        print("2. 'spektral' library is installed and its 'GlobalSumPool' layer is correctly used/imported.")
        print("3. The .h5 file is not corrupted.")
        return []

    print(f"\n--- Starting prediction process for images in: '{prediction_dir_path}' ---")

    # Collect all image paths from the directory (recursively)
    image_paths_to_predict = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for root, _, files in os.walk(prediction_dir_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths_to_predict.append(os.path.join(root, file))

    if not image_paths_to_predict:
        print(f"No valid image files found in '{prediction_dir_path}'. Please check the directory content and file extensions.")
        return []

    # Initialize MediaPipe Hands model once for efficiency
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1, # Assuming one hand per image, consistent with training
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence # Used for static_image_mode
    )

    all_landmarks_for_prediction = []
    original_image_paths = [] # To keep track of which image path corresponds to which landmarks
    skipped_images_count = 0
    total_images_found = len(image_paths_to_predict)

    print(f"Found {total_images_found} images. Extracting hand landmarks...")
    for img_path in tqdm(image_paths_to_predict, desc="Extracting landmarks"):
        landmarks = extract_landmarks(img_path, mp_hands, image_size=image_size)
        if landmarks is not None:
            all_landmarks_for_prediction.append(landmarks)
            original_image_paths.append(img_path)
        else:
            skipped_images_count += 1
            # print(f"  Skipped: {os.path.basename(img_path)} (no hand detected or error)")

    mp_hands.close() # Close MediaPipe instance after all extractions
    print(f"Landmark extraction complete. Skipped {skipped_images_count} images (no hand detected or processing error).")

    if not all_landmarks_for_prediction:
        print("No hand landmarks could be extracted from any image. Cannot perform prediction.")
        return []

    # Prepare inputs for the model in batch format
    # x_batch: (num_images, num_nodes, features) -> (N, 21, 3)
    x_batch = np.array(all_landmarks_for_prediction, dtype=np.float32)
    # a_batch: (num_images, num_nodes, num_nodes) -> (N, 21, 21)
    # Adjacency matrix is constant for all hands in this model, so tile it.
    a_batch = np.tile(_create_hand_adjacency(), (x_batch.shape[0], 1, 1)).astype(np.float32)

    print(f"Performing batch prediction on {x_batch.shape[0]} images...")
    start_time = time.time()
    # Perform prediction with a verbose progress bar
    predictions = model.predict([x_batch, a_batch], batch_size=batch_size, verbose=1)
    end_time = time.time()
    print(f"Batch prediction took {end_time - start_time:.4f} seconds.")

    print(f"\n--- Prediction Results ---")
    prediction_results = []
    for i, pred_probs in enumerate(predictions):
        predicted_class_idx = np.argmax(pred_probs)
        confidence = pred_probs[predicted_class_idx]

        # Map index to class name
        predicted_class_name = "UNKNOWN_CLASS"
        if predicted_class_idx < len(class_names):
            predicted_class_name = class_names[predicted_class_idx]
        else:
            print(f"Warning: Predicted class index {predicted_class_idx} is out of bounds for the provided CLASS_NAMES list. Please check CLASS_NAMES order and length.")

        result = {
            "image_path": original_image_paths[i],
            "predicted_sign": predicted_class_name,
            "confidence": confidence
        }
        prediction_results.append(result)

        # Print results clearly
        print(f"Image: {os.path.basename(original_image_paths[i]):<35} -> Predicted: {predicted_class_name:<20} | Confidence: {confidence*100:.2f}%")

    print(f"\n--- Prediction Summary ---")
    print(f"Total images found in directory: {total_images_found}")
    print(f"Images successfully processed and predicted: {len(original_image_paths)}")
    print(f"Images skipped (no hand detected or error): {skipped_images_count}")

    return prediction_results

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration Variables (YOU MUST EDIT THESE) ---

    # Path to your saved .h5 model file
    MODEL_PATH = "final_isl_gnn_model_high_accuracy.h5" # Or "best_isl_gnn_model.h5"

    # Directory containing the images you want to predict.
    # The script will search for images recursively within this directory.
    PREDICTION_DIR_PATH = "G:/Sign_language_GNN/model/imagedata-main" # <--- IMPORTANT: Change this path

    # The list of class names IN THE EXACT ORDER they were used during training.
    # This list was likely printed by your training script.
    CLASS_NAMES = [
        '0','1','2','A','B','C'
    ]


    # --- GPU Configuration (Recommended) ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set TensorFlow to use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            # Allow memory growth to prevent pre-allocation of all GPU memory
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"🔥 TensorFlow is configured to use GPU: {gpus[0].name}")
        except RuntimeError as e:
            # Handle error if GPU setting fails
            print(f"Error configuring GPU: {e}")
            print("Falling back to CPU if no other GPU is available or configured.")
    else:
        print("⚠️ No GPU detected. TensorFlow will use CPU for prediction, which might be slower.")

    # --- Run Prediction ---
    if os.path.exists(MODEL_PATH) and os.path.isdir(PREDICTION_DIR_PATH) and len(CLASS_NAMES) > 0:
        print("\nStarting ISL Sign Prediction...")
        results = run_predictions_on_directory(MODEL_PATH, PREDICTION_DIR_PATH, CLASS_NAMES)

        if results:
            print("\nPrediction process completed successfully.")
            # Optional: Save results to a CSV file for detailed analysis
            try:
                df = pd.DataFrame(results)
                output_csv_path = "isl_prediction_results.csv"
                df.to_csv(output_csv_path, index=False)
                print(f"All prediction results saved to '{output_csv_path}'")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")
        else:
            print("No predictions were made. Please check the provided paths and ensure images contain detectable hands.")
    else:
        print("\nConfiguration Error: Please fix the paths or CLASS_NAMES in the script before running.")
        if not os.path.exists(MODEL_PATH):
            print(f"Model path error: '{MODEL_PATH}' does not exist.")
        if not os.path.isdir(PREDICTION_DIR_PATH):
            print(f"Prediction directory error: '{PREDICTION_DIR_PATH}' does not exist or is not a directory.")
        if not CLASS_NAMES:
            print("CLASS_NAMES list is empty. It must contain your trained class labels.")