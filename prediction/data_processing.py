import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import math

class ISLDataset:
    def __init__(self, dataset_path, img_size=(224, 224), validation_split=0.2, test_split=0.1, random_state=42):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_state = random_state

        # Sort classes alphabetically to ensure consistent indexing
        self.classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

        self.mp_hands = mp.solutions.hands
        self.all_data = []

        if not self.classes:
            raise ValueError(f"No class subdirectories found in {dataset_path}. Please organize your dataset into subfolders, e.g., 'A/', 'B/', etc.")

    def _augment_landmarks(self, landmarks):
        """
        Applies various augmentations to 3D hand landmarks.
        `landmarks` is a flat array of 63 values (21 * 3).
        """
        augmented_landmarks = landmarks.reshape(-1, 3).copy() # Reshape to (21, 3)

        # 1. Random Scaling (around a central point, e.g., wrist 0)
        scale_factor = np.random.uniform(0.9, 1.1) # Scale between 90% and 110%
        # Translate to origin, scale, then translate back
        center_point = augmented_landmarks[0] # Using wrist as origin for scaling/rotation
        augmented_landmarks = (augmented_landmarks - center_point) * scale_factor + center_point

        # 2. Random Rotation (around Z-axis for 2D/3D planar rotation, adjust if full 3D rotation needed)
        angle_degrees = np.random.uniform(-15, 15) # Rotate +/- 15 degrees
        angle_rad = math.radians(angle_degrees)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        augmented_landmarks = np.dot(augmented_landmarks - center_point, rotation_matrix) + center_point

        # 3. Random Translation (shift) - apply to normalized coordinates
        # Since landmarks are normalized (0-1), a small shift is appropriate
        shift_range = 0.05
        shift_x = np.random.uniform(-shift_range, shift_range)
        shift_y = np.random.uniform(-shift_range, shift_range)
        shift_z = np.random.uniform(-shift_range, shift_range) # Less impactful for 2D projections
        augmented_landmarks[:, 0] += shift_x
        augmented_landmarks[:, 1] += shift_y
        augmented_landmarks[:, 2] += shift_z

        # 4. Add small random noise
        noise_level = 0.005 # Small Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_level, size=augmented_landmarks.shape)
        augmented_landmarks += noise

        return augmented_landmarks.flatten() # Flatten back to 1D

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1, # Focus on single hand for ISL letters typically
            min_detection_confidence=0.7
        ) as hands:
            for class_idx, class_name in enumerate(tqdm(self.classes, desc="Processing classes")):
                class_path = os.path.join(self.dataset_path, class_name)
                if not os.path.isdir(class_path):
                    print(f"Warning: Class directory {class_path} not found. Skipping.")
                    continue

                valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(valid_extensions)]

                if not image_files:
                    print(f"Warning: No valid image files found in {class_path}. Skipping.")
                    continue

                for img_name in image_files:
                    img_path = os.path.join(class_path, img_name)

                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Warning: Could not read image {img_path}. Skipping.")
                            continue

                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize for consistent MediaPipe processing, but keep aspect ratio or pad if needed
                        # For now, simple resize that might distort aspect, but MediaPipe might be robust
                        img_resized = cv2.resize(img_rgb, self.img_size, interpolation=cv2.INTER_AREA)

                        results = hands.process(img_resized)

                        if results.multi_hand_landmarks:
                            # We take the first hand detected, as ISL often uses one hand per sign
                            hand_landmarks = results.multi_hand_landmarks[0]
                            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                            self.all_data.append({
                                'landmarks': landmarks,
                                'label': class_idx
                            })

                            # Add augmented data (e.g., 3 augmented versions per original image)
                            for _ in range(3): # You can adjust the number of augmentations
                                self.all_data.append({
                                    'landmarks': self._augment_landmarks(landmarks),
                                    'label': class_idx
                                })
                        # else: # Optional: Log images where no hand is detected
                            # print(f"No hand detected in {img_path}")

                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue

        if not self.all_data:
            raise ValueError("No data was loaded. Check dataset path, image files, and MediaPipe's ability to detect hands.")

        # Save class names to a file for use in the prediction script
        classes_file_path = "classes.txt"
        with open(classes_file_path, 'w') as f:
            for cls_name in self.classes:
                f.write(f"{cls_name}\n")
        print(f"✅ Class names saved to {classes_file_path}")

        return self._prepare_graph_data()

    def _prepare_graph_data(self):
        num_landmarks = 21
        num_features_per_landmark = 3

        landmarks_list = [d['landmarks'].reshape(num_landmarks, num_features_per_landmark) for d in self.all_data]
        labels_list = [d['label'] for d in self.all_data]

        X = np.array(landmarks_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int32)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
            (5, 9), (9, 13), (13, 17) # Connecting bases of fingers (metacarpals)
        ]

        A = np.zeros((len(self.all_data), num_landmarks, num_landmarks), dtype=np.float32)
        for i in range(len(self.all_data)):
            adj_matrix = np.eye(num_landmarks, dtype=np.float32) # Add self-loops
            for u, v in connections:
                adj_matrix[u, v] = 1.0
                adj_matrix[v, u] = 1.0
            A[i] = adj_matrix

        # Split data
        X_train_val, X_test, A_train_val, A_test, y_train_val, y_test = train_test_split(
            X, A, y, test_size=self.test_split, random_state=self.random_state, stratify=y
        )

        validation_size_relative_to_train_val = self.validation_split / (1 - self.test_split)

        X_train, X_val, A_train, A_val, y_train, y_val = train_test_split(
            X_train_val, A_train_val, y_train_val,
            test_size=validation_size_relative_to_train_val,
            random_state=self.random_state, stratify=y_train_val
        )

        print(f"\nDataset loaded. Total samples (including augmented): {len(self.all_data)}")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {len(self.classes)}")

        return (X_train, A_train, y_train), \
               (X_val, A_val, y_val), \
               (X_test, A_test, y_test)

if __name__ == '__main__':
    dataset_path = '/content/sign_image/imagedata-main'

    try:
        dataset = ISLDataset(dataset_path)
        (X_train, A_train, y_train), (X_val, A_val, y_val), (X_test, A_test, y_test) = dataset.load_and_preprocess_data()

        print("\nShapes of the preprocessed data:")
        print(f"X_train shape: {X_train.shape}")
        print(f"A_train shape: {A_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Classes found: {dataset.classes}")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure 'ISL_Dataset' exists and contains subfolders with image files.")
    except FileNotFoundError:
        print(f"Error: Dataset path '{dataset_path}' not found.")
        print("Please ensure the dataset path is correct and the directory exists.")