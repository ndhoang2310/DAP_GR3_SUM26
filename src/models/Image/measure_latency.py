import sys
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))

import config


NUM_SAMPLES = 100


def main():
    processed_dir = Path(config.CNN_PROCESSED_DIR)
    model_dir = Path(config.MODEL_SAVE_PATH)

    X_test_img = np.load(processed_dir / "X_test_img.npy")[:NUM_SAMPLES]
    X_test_cnn = np.load(processed_dir / "X_test_cnn.npy")[:NUM_SAMPLES]

    svm_model = joblib.load(model_dir / "svm_model.pkl")

    cnn_model_path = model_dir / "best_cnn.keras"
    if not cnn_model_path.exists():
        cnn_model_path = model_dir / "cnn_blink_model.keras"
    cnn_model = load_model(cnn_model_path)

    print("===== LATENCY MEASUREMENT =====")
    print(f"Samples: {len(X_test_img)}")

    # SVM latency
    start = time.perf_counter()
    for i in range(len(X_test_img)):
        _ = svm_model.predict(X_test_img[i:i + 1])
    svm_latency_ms = (time.perf_counter() - start) / len(X_test_img) * 1000

    # CNN warm-up
    warmup = tf.convert_to_tensor(X_test_cnn[:1])
    _ = cnn_model(warmup, training=False)

    start = time.perf_counter()
    for i in range(len(X_test_cnn)):
        sample = tf.convert_to_tensor(X_test_cnn[i:i + 1])
        _ = cnn_model(sample, training=False)
    cnn_latency_ms = (time.perf_counter() - start) / len(X_test_cnn) * 1000

    print(f"SVM Latency: {svm_latency_ms:.4f} ms/image")
    print(f"CNN Latency: {cnn_latency_ms:.4f} ms/image")


if __name__ == "__main__":
    main()
