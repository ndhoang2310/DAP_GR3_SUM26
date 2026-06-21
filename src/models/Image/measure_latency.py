import numpy as np
import joblib
import time
from pathlib import Path
from tensorflow.keras.models import load_model
import sys

# Setup path
base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))
import config

def measure_latency():
    # Load data
    processed_dir = Path(config.CNN_PROCESSED_DIR)
    X_test_cnn = np.load(processed_dir / "X_test_cnn.npy")[:100] # Test 100 mẫu
    X_test_img = np.load(processed_dir / "X_test_img.npy")[:100]

    # Load models
    svm_model = joblib.load(Path(config.MODEL_SAVE_PATH) / "svm_model.pkl")
    cnn_model = load_model(Path(config.MODEL_SAVE_PATH) / "cnn_blink_model.keras")

    # Đo SVM
    start = time.perf_counter()
    for i in range(len(X_test_img)):
        svm_model.predict(X_test_img[i:i+1])
    svm_latency = (time.perf_counter() - start) / len(X_test_img) * 1000

    # Đo CNN
    # Lần đầu CNN chạy thường chậm do khởi tạo, ta bỏ qua lần dự đoán đầu
    cnn_model.predict(X_test_cnn[:1], verbose=0) 
    
    start = time.perf_counter()
    for i in range(len(X_test_cnn)):
        cnn_model.predict(X_test_cnn[i:i+1], verbose=0)
    cnn_latency = (time.perf_counter() - start) / len(X_test_cnn) * 1000

    print(f"--- ĐỘ TRỄ TRUNG BÌNH (Mỗi ảnh) ---")
    print(f"SVM Latency: {svm_latency:.4f} ms")
    print(f"CNN Latency: {cnn_latency:.4f} ms")

if __name__ == "__main__":
    measure_latency()