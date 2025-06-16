# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import mediapipe as mp

# Load the pre-trained model
model = load_model('gnn_model.h5')
print("Model loaded successfully.")

# Define class names (0-9 and A-Z)
class_names = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Function to remove background and focus on the hand
def remove_background(frame, landmarks):
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Get hand landmarks and create a convex hull
    points = []
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        points.append((x, y))
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply the mask to the frame
    hand_only = cv2.bitwise_and(frame, frame, mask=mask)
    return hand_only

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

            # Crop the hand region
            hand_image = frame[y_min:y_max, x_min:x_max]

            # Remove background from the hand region
            hand_only = remove_background(frame, hand_landmarks.landmark)
            hand_only_cropped = hand_only[y_min:y_max, x_min:x_max]

            # Resize the hand-only image to match the model's expected input size
            if hand_only_cropped.size != 0:
                hand_image_resized = cv2.resize(hand_only_cropped, (150, 150))
                hand_image_array = image.img_to_array(hand_image_resized)
                hand_image_array = np.expand_dims(hand_image_array, axis=0)
                hand_image_array /= 255.0  # Normalize the image

                # Make the prediction
                prediction = model.predict(hand_image_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Draw the bounding box and prediction on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Predicted: {predicted_class} ({confidence:.2f})",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the hand-only region
                cv2.imshow('Hand Only (Background Removed)', hand_only_cropped)

    # Show the frame with predictions
    cv2.imshow('Webcam Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()