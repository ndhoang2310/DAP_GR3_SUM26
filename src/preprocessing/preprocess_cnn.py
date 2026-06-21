import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Nạp đường dẫn gốc để import config
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(base_dir)
import config

def load_and_process_images():
    csv_path = os.path.join(config.MASTER_DATASET_DIR, "metadata_master.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file metadata tại: {csv_path}")

    df = pd.read_csv(csv_path)
    # Lọc các dòng thành công và nhãn hợp lệ
    df = df[(df["status"] == "success") & (df["final_label"].isin(["open", "closed"]))].copy()

    X, y = [], []
    
    print("Đang đọc và xử lý ảnh cho CNN...")
    for _, row in df.iterrows():
        image_path = row["image_path"]
        
        if not os.path.exists(image_path):
            continue
            
        # Đọc ảnh xám trực tiếp
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Resize về 24x24
        img = cv2.resize(img, config.IMAGE_SIZE)
        # Chuẩn hóa pixel về [0, 1] và thêm chiều channel (24, 24, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1) 
        
        X.append(img)
        label = 0 if row["final_label"] == "open" else 1
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    return X, y

def main():
    X, y = load_and_process_images()
    print(f"Tổng số mẫu hợp lệ: {len(X)}, Shape của dữ liệu ảnh: {X.shape}")

    # Chia tập train / test
    # ==================================================
    # Bước 1:
    # tạo test giống SVM
    # ==================================================

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==================================================
    # Bước 2:
    # chia train-val cho CNN
    # ==================================================

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=42,
        stratify=y_trainval
    )
    # Lưu numpy array vào thư mục processed_image
    os.makedirs(config.CNN_PROCESSED_DIR, exist_ok=True)
    
    np.save(os.path.join(config.CNN_PROCESSED_DIR, "X_train_cnn.npy"), X_train)
    np.save(os.path.join(config.CNN_PROCESSED_DIR, "y_train_cnn.npy"), y_train)

    np.save(os.path.join(config.CNN_PROCESSED_DIR, "X_val_cnn.npy"), X_val)
    np.save(os.path.join(config.CNN_PROCESSED_DIR, "y_val_cnn.npy"), y_val)

    np.save(os.path.join(config.CNN_PROCESSED_DIR, "X_test_cnn.npy"), X_test)
    np.save(os.path.join(config.CNN_PROCESSED_DIR, "y_test_cnn.npy"), y_test)
    
    print("Tiền xử lý hoàn tất! Đã lưu các file .npy tại:", config.CNN_PROCESSED_DIR)

if __name__ == "__main__":
    main()