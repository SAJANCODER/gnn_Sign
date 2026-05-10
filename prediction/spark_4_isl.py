import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
from collections import Counter, deque

# ── TTS IMPORTS ──────────────────────────────────────────────────────────────
from gtts import gTTS
import pygame
import threading
import queue
import tempfile

# Custom GNN layers (Make sure model_utils.py is in the same directory)
from model_utils import CustomGCNConv, GlobalSumPool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
#  🔊  ROBUST NEURAL TTS ENGINE
# =============================================================================

speech_queue = queue.Queue()

# Initialize pygame mixer for smooth background audio
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=2048)
pygame.mixer.init()
pygame.init()


def _speak_now(text: str) -> None:
    """Generates and plays human-sounding neural audio."""
    tmp_path = None
    try:
        # Using Indian English neural voice (tld='co.in')
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, prefix='isl_tts_') as tmp:
            tmp_path = tmp.name
        tts.save(tmp_path)

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()

    except OSError as net_err:
        print(f"\n🚫 TTS network error — check your internet connection.\n   Detail: {net_err}\n")
    except Exception as exc:
        print(f"\n🚫 TTS unexpected error: {exc}\n")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def tts_worker() -> None:
    """Background daemon thread to handle audio queuing."""
    while True:
        text = speech_queue.get()
        if text is None:
            speech_queue.task_done()
            break
        if text.strip():
            _speak_now(text.strip())
        speech_queue.task_done()


# Start daemon thread immediately
_tts_thread = threading.Thread(target=tts_worker, daemon=True, name='TTS-Worker')
_tts_thread.start()


# =============================================================================
#  MODEL + MEDIAPIPE LOGIC
# =============================================================================

def load_inference_model(model_path, classes_file='classes.txt'):
    custom_objects = {'CustomGCNConv': CustomGCNConv, 'GlobalSumPool': GlobalSumPool}
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"🚫 Error loading model: {e}")
        return None, None

    classes = []
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"✅ Classes loaded: {classes}")
        except Exception as e:
            print(f"🚫 Error loading classes: {e}")
            return None, None
    return model, classes


def preprocess_frame(frame, mp_hands):
    """Extracts landmarks and isolates the largest hand in the frame."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = mp_hands.process(image_rgb)
    image_rgb.flags.writeable = True

    landmarks = None
    main_hand_landmarks = None

    if results.multi_hand_landmarks:
        best_area = 0
        # Spatial Filter: Find the hand closest to the camera (largest bounding box)
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

            if area > best_area:
                best_area = area
                main_hand_landmarks = hand_landmarks

        # Flatten the targeted hand into the (1, 21, 3) matrix required by the GNN
        if main_hand_landmarks:
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in main_hand_landmarks.landmark]
            ).flatten().reshape(1, 21, 3).astype(np.float32)

    return landmarks, results, main_hand_landmarks


def create_adjacency_matrix(num_landmarks=21):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
    ]
    adj = np.eye(num_landmarks, dtype=np.float32)
    for u, v in connections:
        adj[u, v] = adj[v, u] = 1.0
    return adj[np.newaxis, :, :]


def visualize_landmarks(image, all_hand_landmarks, main_hand, mp_drawing, mp_hand_connections):
    """
    🔥 DYNAMIC HIGHLIGHTING LOGIC
    Draws the focused hand in bold RED, and background hands in standard GREEN.
    """
    if all_hand_landmarks:
        for lm in all_hand_landmarks:
            is_main = False

            # Check if this hand's wrist coordinate matches our isolated main_hand
            if main_hand and (
                    lm.landmark[0].x == main_hand.landmark[0].x and lm.landmark[0].y == main_hand.landmark[0].y):
                is_main = True

            if is_main:
                # Active Focus Hand: Thicker RED lines (BGR format: 0, 0, 255)
                dot_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
                line_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2)
            else:
                # Background Hands: Standard GREEN lines (BGR format: 0, 255, 0)
                dot_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

            mp_drawing.draw_landmarks(
                image, lm, mp_hand_connections.HAND_CONNECTIONS,
                dot_spec, line_spec
            )
    return image


def detect_swipe(wrist_history):
    """Heuristic UI controls based on wrist movement over 10 frames."""
    if len(wrist_history) < 10:
        return None
    dx = wrist_history[-1][0] - wrist_history[0][0]
    dy = wrist_history[-1][1] - wrist_history[0][1]
    SWIPE_THRESHOLD = 0.25

    if dx < -SWIPE_THRESHOLD and abs(dy) < 0.2:
        return "BACKSPACE"
    if dy > SWIPE_THRESHOLD and abs(dx) < 0.2:
        return "CLEAR"
    if dy < -SWIPE_THRESHOLD and abs(dx) < 0.2:
        return "SPEAK"
    return None


# =============================================================================
#  MAIN LOOP
# =============================================================================

def main():
    model_path = 'final_isl_gnn_model_full.h5'
    classes_file = 'classes_1.txt'

    model, classes = load_inference_model(model_path, classes_file)
    if model is None:
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fixed_A = create_adjacency_matrix()

    CONFIDENCE_THRESHOLD = 0.70
    BUFFER_SIZE = 10
    PAUSE_THRESHOLD = 15

    prediction_buffer = deque(maxlen=BUFFER_SIZE)
    wrist_history = deque(maxlen=15)

    pause_counter = 0
    current_word = ""
    last_added_letter = ""

    print("\n🚀 ISL Recognition Engine started.  Press 'q' to quit.")
    print("   Swipe LEFT → Backspace | Swipe DOWN → Clear | Swipe UP → Speak\n")

    with mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2,  # Allows multiple hands to test the red/green highlighting
    ) as hands:

        prev_frame_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, results, main_hand = preprocess_frame(frame, hands)

            display_text = "No Hand Detected"

            if landmarks is not None and main_hand is not None:
                wrist_x = main_hand.landmark[0].x
                wrist_y = main_hand.landmark[0].y
                wrist_history.append((wrist_x, wrist_y))

                swipe_action = detect_swipe(wrist_history)

                if swipe_action == "BACKSPACE":
                    current_word = current_word[:-1]
                    last_added_letter = ""
                    wrist_history.clear()
                    prediction_buffer.clear()
                    display_text = "Action: BACKSPACE"

                elif swipe_action == "CLEAR":
                    current_word = ""
                    last_added_letter = ""
                    wrist_history.clear()
                    prediction_buffer.clear()
                    display_text = "Action: CLEAR WORD"

                elif swipe_action == "SPEAK":
                    word = current_word.strip()
                    if word:
                        print(f"\n🔊 Speaking: '{word}'")
                        speech_queue.put(word)
                        display_text = f"Speaking: {word}"
                    wrist_history.clear()
                    prediction_buffer.clear()
                    last_added_letter = ""

                else:
                    preds = model.predict([landmarks, fixed_A], verbose=0)
                    predicted_idx = np.argmax(preds[0])
                    confidence = preds[0][predicted_idx]

                    if classes and confidence > CONFIDENCE_THRESHOLD:
                        predicted_name = classes[predicted_idx]
                        display_text = f"{predicted_name} ({confidence * 100:.1f}%)"

                        prediction_buffer.append(predicted_name)
                        pause_counter = 0

                        # Temporal Smoothing: Require N consecutive frames to confirm a sign
                        if len(prediction_buffer) == BUFFER_SIZE:
                            most_common, count = Counter(prediction_buffer).most_common(1)[0]
                            if count >= int(BUFFER_SIZE * 0.7) and most_common != last_added_letter:
                                current_word += most_common
                                last_added_letter = most_common
                                prediction_buffer.clear()
                    else:
                        display_text = "Uncertain / No Clear Sign"
                        pause_counter += 1
                        prediction_buffer.clear()
            else:
                pause_counter += 1
                prediction_buffer.clear()
                wrist_history.clear()

            # ── Auto-space on pause
            if pause_counter > PAUSE_THRESHOLD:
                if current_word and current_word[-1] != " ":
                    current_word += " "
                last_added_letter = ""
                pause_counter = PAUSE_THRESHOLD + 1

                # ── Draw landmarks with Focus Highlighting
            frame = visualize_landmarks(
                frame, results.multi_hand_landmarks, main_hand, mp_drawing, mp_hands
            )

            # ── FPS
            now = time.time()
            fps = 1 / max(now - prev_frame_time, 1e-6)
            prev_frame_time = now

            # ── UI Overlay
            h, w = frame.shape[:2]
            cv2.putText(frame, f"FPS: {int(fps)}",
                        (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Sign: {display_text}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Text: {current_word}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            cv2.imshow('ISL Recognition Engine', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Speak final accumulated text before closing
    final = current_word.strip()
    if final:
        print(f"\n🔊 Speaking final output: '{final}'")
        speech_queue.put(final)
        speech_queue.join()

        # ── Shutdown TTS thread cleanly
    speech_queue.put(None)
    _tts_thread.join(timeout=5)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Session ended.  Final text: '{final}'")


# =============================================================================
if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    main()