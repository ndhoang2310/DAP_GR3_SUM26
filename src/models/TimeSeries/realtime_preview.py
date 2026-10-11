"""
realtime_preview.py – Real-time evaluation of the TimeSeries EAR models on webcam or video files.

Updates:
    - Fixed single-eye occlusion bug: Changed EAR calculation from average to min(ear_left, ear_right).
      When one eye is covered/occluded, MediaPipe often predicts a fake "open" state for it.
      By taking the minimum of the two eyes, we track the active eye's blinks and closures,
      making the system robust to hand occlusion, winking, and profile views.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import joblib
from collections import deque

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try importing MediaPipe
try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe is not installed. Install it with: pip install mediapipe")
    sys.exit(1)

# EAR landmark indices
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

def compute_ear(eye_points):
    """Compute Eye Aspect Ratio (EAR) given 6 landmarks."""
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

def extract_advanced_features_single(w):
    """Extract 28 advanced dynamic features from a 7-frame EAR window."""
    min_val = np.min(w)
    max_val = np.max(w)
    mean_val = np.mean(w)
    std_val = np.std(w)
    rng_val = max_val - min_val
    
    drop_left = w[0] - w[3]
    drop_right = w[6] - w[3]
    ratio_center = (w[0] + w[6]) / (2 * w[3] + 1e-5)
    
    diff1 = np.diff(w)
    diff2 = np.diff(diff1)
    
    # --- New Optimized Features ---
    if std_val > 1e-5:
        kurtosis = np.mean(((w - mean_val) / std_val) ** 4) - 3.0
    else:
        kurtosis = 0.0
    slope_trend = (w[-1] - w[0]) / 6.0
    
    feat = np.concatenate([
        w,
        [min_val, max_val, mean_val, std_val, rng_val, drop_left, drop_right, ratio_center],
        diff1,
        diff2,
        [kurtosis, slope_trend]
    ])
    return feat

def extract_advanced_12_features_single(w):
    """Trích xuất 12 đặc trưng động học độc lập (loại bỏ hoàn toàn collinearity)."""
    min_val = np.min(w)
    max_val = np.max(w)
    std_val = np.std(w)
    ratio_center = (w[0] + w[6]) / (2 * w[3] + 1e-5)
    mean_val = np.mean(w)
    if std_val > 1e-5:
        kurtosis = np.mean(((w - mean_val) / std_val) ** 4) - 3.0
    else:
        kurtosis = 0.0
        
    feat = np.concatenate([
        w,
        [min_val, max_val, std_val, ratio_center, kurtosis]
    ])
    return feat

def main():
    parser = argparse.ArgumentParser(description="Real-time TimeSeries EAR model preview.")
    parser.add_argument(
        "--input", type=str, default="0",
        help="Path to video file or webcam index (default: 0)"
    )
    parser.add_argument(
        "--model", type=str, default=os.path.join("dataset_master", "models", "best_traditional_model.pkl"),
        help="Path to trained model pickle (default: dataset_master/models/best_traditional_model.pkl)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Hybrid filter threshold (default: 0.2)"
    )
    args = parser.parse_args()

    # Load model
    print(f"[INFO] Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found at {args.model}.")
        sys.exit(1)
    
    try:
        model = joblib.load(args.model)
        print(f"[SUCCESS] Model loaded: {type(model)}")
        
        # Determine model tag dynamically for HUD display
        if hasattr(model, "steps"):  # It's a scikit-learn Pipeline
            underlying_model = model.steps[-1][1]
        else:
            underlying_model = model
        
        model_name_class = type(underlying_model).__name__
        if "RandomForest" in model_name_class:
            model_tag = "RF"
        elif "GradientBoosting" in model_name_class:
            model_tag = "GB"
        elif "SVC" in model_name_class:
            model_tag = "SVM"
        elif "KNeighbors" in model_name_class:
            model_tag = "KNN"
        else:
            model_tag = "ML"
        print(f"[INFO] Detected model type: {model_tag} ({model_name_class})")
        # Determine number of input features expected by the model
        if hasattr(model, "n_features_in_"):
            num_features = model.n_features_in_
        elif hasattr(underlying_model, "n_features_in_"):
            num_features = underlying_model.n_features_in_
        else:
            num_features = 28
        print(f"[INFO] Model expects {num_features} input features.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open video source
    video_source = args.input
    if video_source.isdigit():
        video_source = int(video_source)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        sys.exit(1)

    # Window sizes and histories
    window_size = 7
    ear_history = deque(maxlen=window_size)
    ear_history_long = deque(maxlen=15) # Keep a longer history for peak/valley detection
    pred_history = deque(maxlen=15) # Extended history for better temporal smoothing (approx 1 second at 15 FPS)
    
    # Dynamic Calibration parameters (expanding_max)
    calibration_frames = 15
    calibration_ears = []
    calib_baseline = 0.25
    is_calibrated = False
    
    # Persistent status state to prevent UI flickering
    current_status = "Calibrating..."
    color_status = (0, 255, 255)
    cpu_mode = "N/A"
    
    # State tracking
    blink_count = 0
    frame_counter = 0
    counted_blink_frames = set()
    
    # FPS Measurement & Time-based Sampling
    start_fps_time = time.time()
    frame_count_fps = 0
    actual_fps = 0.0
    
    # 15 FPS downsampling configuration
    target_sampling_fps = 15.0
    sampling_interval = 1.0 / target_sampling_fps # 0.0667 seconds
    last_sample_time = time.time()
    
    # CPU savings statistics
    total_windows_processed = 0
    skipped_ml_count = 0
    
    print("=" * 60)
    print("   REAL-TIME TIME-SERIES EAR PREVIEW TOOL   ")
    print("=" * 60)
    print("  Controls:")
    print("    'r' - Reset calibration")
    print("    'q' - Quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or cannot read frame.")
            break

        # Calculate actual camera preview FPS
        frame_count_fps += 1
        elapsed = time.time() - start_fps_time
        if elapsed >= 1.0:
            actual_fps = frame_count_fps / elapsed
            frame_count_fps = 0
            start_fps_time = time.time()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. MediaPipe tracking
        results = face_mesh.process(rgb)
        
        ear_avg = None
        ear_norm = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            pts = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            
            # Extract landmarks for eyes
            left_eye_pts = pts[LEFT_EYE_EAR_IDX]
            right_eye_pts = pts[RIGHT_EYE_EAR_IDX]
            
            # Compute raw EAR for both eyes
            ear_left = compute_ear(left_eye_pts)
            ear_right = compute_ear(right_eye_pts)
            
            # --- FIX: ROBUST SINGLE-EYE OCCLUSION HANDLING ---
            # Instead of taking the average, we take the minimum of both eyes.
            ear_avg = min(ear_left, ear_right)
            
            # Draw eye landmarks on screen
            for pt in np.vstack([left_eye_pts, right_eye_pts]):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
            
            # 2. Dynamic calibration of EAR baseline (expanding_max)
            if not is_calibrated:
                calibration_ears.append(ear_avg)
                current_status = "Initializing..."
                color_status = (0, 255, 255)
                
                if len(calibration_ears) >= calibration_frames:
                    calib_baseline = np.max(calibration_ears)
                    if np.isnan(calib_baseline) or calib_baseline < 0.18:
                        calib_baseline = 0.23
                    is_calibrated = True
                    print(f"\n[CALIBRATION COMPLETE] calib_baseline (init max): {calib_baseline:.4f}")
            else:
                # Update baseline as expanding max
                if ear_avg > calib_baseline:
                    calib_baseline = ear_avg
                
                # Normalize current EAR using expanding max
                ear_norm = ear_avg / calib_baseline
                ear_norm = np.clip(ear_norm, 0.0, 1.0)
                
                # --- TIME-BASED DOWNSAMPLING ---
                current_time = time.time()
                if current_time - last_sample_time >= sampling_interval:
                    ear_history.append(ear_norm)
                    ear_history_long.append(ear_norm)
                    frame_counter += 1
                    last_sample_time = current_time
                    
                    # --- VALLEY DETECTION BLINK COUNTING (Protects against double/triple rapid blinks) ---
                    w_len = len(ear_history_long)
                    if w_len >= 3:
                        for i in range(1, w_len - 1):
                            # Look for local minimum (valley) in EAR sequence
                            if ear_history_long[i] < ear_history_long[i-1] and ear_history_long[i] < ear_history_long[i+1]:
                                if ear_history_long[i] < 0.35: # Threshold for eye closure depth
                                    abs_frame = frame_counter - (w_len - 1 - i)
                                    if abs_frame not in counted_blink_frames:
                                        too_close = False
                                        for past_frame in counted_blink_frames:
                                            # Minimum 2 frames distance (~0.13 seconds) to filter noise/multiple peaks in one blink
                                            if abs(abs_frame - past_frame) <= 2: 
                                                too_close = True
                                                break
                                        if not too_close:
                                            blink_count += 1
                                            counted_blink_frames.add(abs_frame)
                                            # Keep the counted set small to avoid memory leak
                                            old_threshold = frame_counter - 100
                                            counted_blink_frames = {f for f in counted_blink_frames if f > old_threshold}
                    
                    # 3. Model Inference (Sliding Window size 7)
                    if len(ear_history) == window_size:
                        total_windows_processed += 1
                        w_arr = np.array(ear_history)
                        
                        # --- HYBRID SYSTEM FILTER ---
                        is_suspicious = np.min(w_arr) < args.threshold
                        
                        if not is_suspicious:
                            pred = 0 
                            skipped_ml_count += 1
                            cpu_mode = "Filter (Skipped ML - 100% Save)"
                        else:
                            if num_features == 28:
                                feat = extract_advanced_features_single(w_arr)
                            elif num_features == 12:
                                feat = extract_advanced_12_features_single(w_arr)
                            else:
                                feat = w_arr
                            pred = model.predict([feat])[0]
                            cpu_mode = f"ML Inference (Active - {model_tag})"
                        
                        pred_history.append(pred)
                        
                        # --- TEMPORAL SMOOTHING & ACTION (15 frames history ~ 1 second) ---
                        is_drowsy = pred_history.count(2) >= 10
                        
                        if is_drowsy:
                            current_status = "WARNING: DROWSY / SLEEPING"
                            color_status = (0, 0, 255) # Red
                        elif pred == 1:
                            current_status = "BLINKING"
                            color_status = (255, 255, 0) # Cyan/Yellow-Green
                        else:
                            current_status = "NORMAL (EYE OPEN)"
                            color_status = (0, 255, 0) # Green
                else:
                    pass
        else:
            current_status = "No Face Detected"
            color_status = (0, 0, 255)
            ear_history.clear()
            pred_history.clear()

        # 4. Render HUD Overlay
        # Background panel for stats
        cv2.rectangle(frame, (10, 10), (320, 240), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 240), (150, 150, 150), 1)
        
        # Display texts
        cv2.putText(frame, "Eye Blink & Drowsiness HUD", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Webcam FPS: {actual_fps:.1f}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Sampling rate: {target_sampling_fps:.1f} FPS", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Raw EAR (Min): {f'{ear_avg:.3f}' if ear_avg is not None else 'N/A'}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Norm EAR: {f'{ear_norm:.3f}' if ear_norm is not None else 'N/A'}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Blinks Counted: {blink_count}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # CPU Saving rate
        save_rate = (skipped_ml_count / max(1, total_windows_processed)) * 100
        cv2.putText(frame, f"CPU Saved: {save_rate:.1f}%", (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if save_rate > 50 else (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"CPU Mode: {cpu_mode}", (20, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw EAR visual bar
        cv2.rectangle(frame, (20, 220), (300, 232), (50, 50, 50), -1)
        if ear_norm is not None:
            bar_w = int(ear_norm * 280)
            cv2.rectangle(frame, (20, 220), (20 + bar_w, 232), (0, 255, 0) if ear_norm >= 0.2 else (0, 0, 255), -1)
        cv2.putText(frame, "EAR Bar (Min)", (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # Status Panel
        status_panel_y = h - 60
        cv2.rectangle(frame, (0, status_panel_y), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, f"STATUS: {current_status}", (20, status_panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_status, 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("TimeSeries EAR Real-Time Preview", frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset calibration
            calibration_ears = []
            is_calibrated = False
            ear_history.clear()
            pred_history.clear()
            current_status = "Calibrating..."
            color_status = (0, 255, 255)
            print("[INFO] Resetting calibration...")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program exited.")

if __name__ == "__main__":
    main()
