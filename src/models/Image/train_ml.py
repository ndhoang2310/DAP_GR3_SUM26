import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_ml():
    # Sử dụng Path để tự động tìm thư mục gốc (DAP_GR3_SUM26)
    # File nằm ở src/models/Image/ nên cần .parent.parent.parent để ra gốc
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    
    # Định nghĩa các đường dẫn chuẩn
    processed_dir = base_dir / "processed_image"
    model_dir = base_dir / "models"
    
    print(f"Đang load dữ liệu từ: {processed_dir}")
    
    # 1. Load dữ liệu
    X_train = np.load(processed_dir / "X_train_img.npy")
    y_train = np.load(processed_dir / "y_train_img.npy")
    X_test = np.load(processed_dir / "X_test_img.npy")
    y_test = np.load(processed_dir / "y_test_img.npy")

    # 2. Khởi tạo và huấn luyện SVM
    print("Đang huấn luyện mô hình SVM...")
    model = SVC(
        kernel='rbf', 
        C=10, 
        gamma='scale', 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # 3. Đánh giá
    y_pred = model.predict(X_test)
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Open', 'Closed']))

    # 4. Lưu mô hình
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "svm_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nMô hình đã được lưu tại: {model_path}")

if __name__ == "__main__":
    train_ml()