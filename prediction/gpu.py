import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import time

# IMPORTANT: Ensure your model_utils.py contains the PyTorch versions
# of CustomGCNConv and GlobalSumPool (inheriting from nn.Module)
from gpu_utils import GNNModel, CustomGCNConv, GlobalSumPool


# Suppress unnecessary logs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_inference_model(model_path, classes_file='classes_1.txt', device='cpu'):
    """
    Loads the trained PyTorch model and class names.
    """
    try:
        # Initialize the architecture (Assuming GNNModel is defined in model_utils)
        model = GNNModel().to(device)

        # Load the weights
        # Note: PyTorch models are typically saved as .pth or .pt
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode

        print(f"✅ PyTorch Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"🚫 Error loading model from {model_path}: {e}")
        return None, None

    # Load class labels
    classes = []
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"✅ Classes loaded: {classes}")
        except Exception as e:
            print(f"🚫 Error loading classes from {classes_file}: {e}")
            return None, None
    else:
        print(f"🚫 Error: {classes_file} not found.")
        return None, None

    return model, classes


def preprocess_frame(frame, mp_hands, device):
    """
    Processes a video frame to extract MediaPipe hand landmarks.
    Returns landmarks as a torch tensor on the specified device.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = mp_hands.process(image_rgb)
    image_rgb.flags.writeable = True

    landmarks_tensor = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        # Convert to torch tensor: (batch_size=1, num_landmarks=21, num_features=3)
        landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)

    return landmarks_tensor, results


def create_adjacency_matrix(device, num_landmarks=21):
    """
    Creates the fixed adjacency matrix for hand landmarks as a torch tensor.
    """
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    adj_matrix = np.eye(num_landmarks, dtype=np.float32)
    for u, v in connections:
        adj_matrix[u, v] = 1.0
        adj_matrix[v, u] = 1.0

    # Return as torch tensor with batch dimension: (1, 21, 21)
    return torch.from_numpy(adj_matrix).unsqueeze(0).to(device)


def visualize_landmarks(image, hand_landmarks, mp_drawing, mp_hand_connections):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                landmarks,
                mp_hand_connections.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
    return image


def main():
    # Set Device for RTX 3050
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"🔥 PyTorch is using GPU: {torch.cuda.get_device_name(0)} for prediction.")
    else:
        print("⚠️ No GPU found, PyTorch will use CPU.")

    model_path = 'final_isl_gnn_model_full.pth'  # Path to your PyTorch .pth file
    classes_file = 'classes_1.txt'

    model, classes = load_inference_model(model_path, classes_file, device)
    if model is None:
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create the fixed adjacency matrix once on the device
    fixed_A = create_adjacency_matrix(device)

    print("\nStarting real-time prediction. Press 'q' to quit.")

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        prev_frame_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            landmarks, results = preprocess_frame(frame, hands, device)

            display_text = "No Hand Detected"

            if landmarks is not None:
                # Disable gradient calculation for inference speed on RTX 3050
                with torch.no_grad():
                    # Conceptually same: model(features, adjacency)
                    output = model(landmarks, fixed_A)

                    # Convert logits to probabilities
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_class_idx = torch.max(probabilities, dim=1)

                    conf_val = confidence.item()
                    idx_val = predicted_class_idx.item()

                CONFIDENCE_THRESHOLD = 0.7
                if classes and conf_val > CONFIDENCE_THRESHOLD:
                    predicted_class_name = classes[idx_val]
                    display_text = f"{predicted_class_name} ({conf_val * 100:.2f}%)"
                else:
                    display_text = "Uncertain"

            frame = visualize_landmarks(frame, results.multi_hand_landmarks, mp_drawing, mp_hands)

            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('ISL Recognition (PyTorch)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()