import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from torch import nn
import torch.nn.functional as F

# === GPU Check ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Define GCN Layer ===
class GCNLayer(nn.Module):
    def init(self, in_channels, out_channels):
        super(GCNLayer, self).init()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, adj):
        deg = torch.sum(adj, dim=-1, keepdim=True)
        deg_inv_sqrt = torch.pow(deg + 1e-10, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(1, 2)
        x = self.linear(x)
        x = torch.bmm(adj_norm, x) + self.bias
        return x

# === Define GCN Model ===
class GCNModel(nn.Module):
    def init(self, in_features, num_classes):
        super(GCNModel, self).init()
        self.gcn1 = GCNLayer(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.gcn2 = GCNLayer(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.gcn3 = GCNLayer(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.4)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.bn1(self.gcn1(x, adj).transpose(1, 2))).transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn2(self.gcn2(x, adj).transpose(1, 2))).transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.bn3(self.gcn3(x, adj).transpose(1, 2))).transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)
        x = torch.sum(x, dim=1)  # Global sum pooling
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.out(x)

# === Define Adjacency Matrix for 21 MediaPipe keypoints ===
def get_fixed_adjacency(num_nodes=21):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    fingers = [
        [0, 1, 2, 3, 4],        # Thumb
        [0, 5, 6, 7, 8],        # Index
        [0, 9,10,11,12],        # Middle
        [0,13,14,15,16],        # Ring
        [0,17,18,19,20]         # Pinky
    ]
    for finger in fingers:
        for i in range(len(finger) - 1):
            A[finger[i], finger[i+1]] = 1
            A[finger[i+1], finger[i]] = 1
    return torch.tensor(A, dtype=torch.float32)

# === Load Model and Classes ===
model_path = "final_isl_gnn_model.pt"
classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Update if different
model = GCNModel(in_features=3, num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
adj = get_fixed_adjacency().unsqueeze(0).to(device)  # (1, 21, 21)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# === Webcam Inference ===
cap = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
            if landmarks.shape[0] == 21:
                x_input = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 21, 3)
                out = model(x_input, adj)
                pred = out.argmax(dim=1).item()
                label = classes[pred]
                cv2.putText(img, f"Predicted: {label}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("ISL Prediction", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
