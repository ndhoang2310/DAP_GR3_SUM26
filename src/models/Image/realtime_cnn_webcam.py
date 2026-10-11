import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# ==========================================================
# IMPORT CONFIG
# ==========================================================

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))

import config
from temporal_smoothing import HysteresisMovingAverageSmoother, TimeBasedClosedAlert


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_NAME = "cnn_blink_model.keras"

# Raw prediction threshold.
# If best_cnn_threshold.txt exists, it will be loaded automatically.
DEFAULT_THRESHOLD = 0.45

# Temporal smoothing setting.
# Hysteresis uses two thresholds to reduce OPEN/CLOSED flickering.
SMOOTH_WINDOW = 9
CLOSE_THRESHOLD = 0.6
OPEN_THRESHOLD = 0.3

# Alert setting.
# Use real time, not frame count, because webcam FPS changes.
ALERT_SECONDS = 1.0

CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
EYE_MARGIN_RATIO = 0.60


# MediaPipe FaceMesh eye landmark indices.
LEFT_EYE_IDX = [
    33, 133, 160, 159, 158, 157, 173,
    246, 161, 163, 144, 145, 153, 154, 155
]

RIGHT_EYE_IDX = [
    362, 263, 387, 386, 385, 384, 398,
    466, 388, 390, 373, 374, 380, 381, 382
]


# ==========================================================
# UTILITIES
# ==========================================================

def load_threshold(processed_dir: Path) -> float:
    threshold_path = processed_dir / "best_cnn_threshold.txt"

    if threshold_path.exists():
        try:
            threshold = float(threshold_path.read_text(encoding="utf-8").strip())
            print(f"Loaded raw threshold from file: {threshold}")
            return threshold
        except ValueError:
            print(f"Invalid threshold file: {threshold_path}. Using default {DEFAULT_THRESHOLD}.")

    print(f"Using default raw threshold: {DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD


def crop_eye(frame: np.ndarray, landmarks, eye_indices, margin_ratio: float = 0.60):
    h, w = frame.shape[:2]

    points = []

    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    points = np.array(points)

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 0 or box_h <= 0:
        return None, None

    margin_x = int(box_w * margin_ratio)
    margin_y = int(box_h * margin_ratio)

    x1 = max(0, x_min - margin_x)
    y1 = max(0, y_min - margin_y)
    x2 = min(w, x_max + margin_x)
    y2 = min(h, y_max + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


def preprocess_eye_for_cnn(eye_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, tuple(config.IMAGE_SIZE))

    normalized = resized.astype(np.float32) / 255.0

    normalized = np.expand_dims(normalized, axis=-1)  # (24, 24, 1)
    normalized = np.expand_dims(normalized, axis=0)   # (1, 24, 24, 1)

    return normalized


def predict_eye_closed_probability(model, eye_bgr: np.ndarray) -> float:
    x = preprocess_eye_for_cnn(eye_bgr)

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)

    # Direct model call is faster than model.predict() for realtime single samples.
    prob = model(x_tensor, training=False).numpy()[0][0]

    return float(prob)


def put_text(frame, text, org, color=(255, 255, 255), scale=0.65, thickness=2):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_eye_boxes(frame, boxes, smooth_pred: int):
    color = (0, 0, 255) if smooth_pred == 1 else (0, 255, 0)

    for box in boxes:
        if box is None:
            continue

        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


# ==========================================================
# MAIN REALTIME LOOP
# ==========================================================

def main():
    processed_dir = Path(config.CNN_PROCESSED_DIR)
    model_path = Path(config.MODEL_SAVE_PATH) / MODEL_NAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    raw_threshold = load_threshold(processed_dir)

    print("===== REALTIME CNN WEBCAM DEMO =====")
    print(f"Model           : {model_path}")
    print(f"Raw threshold   : {raw_threshold}")
    print(f"Smooth window   : {SMOOTH_WINDOW}")
    print(f"Close threshold : {CLOSE_THRESHOLD}")
    print(f"Open threshold  : {OPEN_THRESHOLD}")
    print(f"Alert duration  : {ALERT_SECONDS:.1f}s")
    print("Press 'q' to quit.")

    model = load_model(model_path)

    smoother = HysteresisMovingAverageSmoother(
        window_size=SMOOTH_WINDOW,
        close_threshold=CLOSE_THRESHOLD,
        open_threshold=OPEN_THRESHOLD,
    )

    alert_checker = TimeBasedClosedAlert(
        seconds_required=ALERT_SECONDS,
    )

    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        face_mesh.close()
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    prev_time = time.perf_counter()
    fps = 0.0

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Cannot read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        raw_label_text = "NO FACE"
        smooth_label_text = "NO FACE"
        avg_prob = 0.0
        closed_duration = 0.0
        alert = False
        smooth_pred = 0

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            left_eye, left_box = crop_eye(
                frame,
                landmarks,
                LEFT_EYE_IDX,
                EYE_MARGIN_RATIO,
            )

            right_eye, right_box = crop_eye(
                frame,
                landmarks,
                RIGHT_EYE_IDX,
                EYE_MARGIN_RATIO,
            )

            probs = []
            boxes = []

            if left_eye is not None:
                probs.append(predict_eye_closed_probability(model, left_eye))
                boxes.append(left_box)

            if right_eye is not None:
                probs.append(predict_eye_closed_probability(model, right_eye))
                boxes.append(right_box)

            if probs:
                prob_closed = float(np.mean(probs))

                # Raw prediction is shown for debugging.
                raw_pred = 1 if prob_closed >= raw_threshold else 0
                raw_label_text = "CLOSED" if raw_pred == 1 else "OPEN"

                # Smoothed prediction is used for alerting.
                smooth_pred = smoother.update(prob_closed)
                avg_prob = smoother.get_average_probability()
                smooth_label_text = "CLOSED" if smooth_pred == 1 else "OPEN"

                alert = alert_checker.update(smooth_pred)
                closed_duration = alert_checker.closed_duration

                draw_eye_boxes(frame, boxes, smooth_pred)

            else:
                smoother.reset()
                alert_checker.reset()

        else:
            smoother.reset()
            alert_checker.reset()

        # FPS smoothing.
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now

        if dt > 0:
            current_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

        # ==================================================
        # DISPLAY
        # ==================================================

        put_text(frame, f"Raw: {raw_label_text}", (20, 35), (255, 255, 255))

        smooth_color = (0, 0, 255) if smooth_label_text == "CLOSED" else (0, 255, 0)
        put_text(frame, f"Smooth: {smooth_label_text}", (20, 65), smooth_color)

        put_text(frame, f"Avg prob closed: {avg_prob:.3f}", (20, 95), (255, 255, 255))

        put_text(
            frame,
            f"Closed duration: {closed_duration:.2f}s/{ALERT_SECONDS:.1f}s",
            (20, 125),
            (255, 255, 255),
        )

        put_text(frame, f"FPS: {fps:.1f}", (20, 155), (255, 255, 255))

        if alert:
            put_text(
                frame,
                "ALERT: EYES CLOSED TOO LONG",
                (20, 205),
                (0, 0, 255),
                scale=0.85,
                thickness=3,
            )

        cv2.imshow("Blink Detection - CNN Realtime", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()