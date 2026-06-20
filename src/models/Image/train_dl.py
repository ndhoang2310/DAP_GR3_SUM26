import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

def train_dl():
    # Sử dụng Path để tìm thư mục gốc của dự án
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    
    # Định nghĩa đường dẫn
    processed_dir = base_dir / "processed_image"
    model_dir = base_dir / "models"
    
    print(f"Đang load dữ liệu từ: {processed_dir}")
    
    # 1. Load dữ liệu
    X_train = np.load(processed_dir / "X_train_img.npy")
    y_train = np.load(processed_dir / "y_train_img.npy")
    X_test = np.load(processed_dir / "X_test_img.npy")
    y_test = np.load(processed_dir / "y_test_img.npy")

    # 2. Xây dựng kiến trúc mô hình (MLP vì dữ liệu đầu vào là vector HOG 1D)
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)), # Lấy số lượng đặc trưng từ shape dữ liệu
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3), # Thêm dropout để tránh overfitting
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Huấn luyện
    print("Đang huấn luyện mô hình Deep Learning...")
    model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=20, 
        batch_size=32,
        verbose=1
    )
    
    # 4. Lưu mô hình
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "cnn_model.keras"
    model.save(model_path)
    print(f"\nMô hình đã được lưu tại: {model_path}")

if __name__ == "__main__":
    train_dl()