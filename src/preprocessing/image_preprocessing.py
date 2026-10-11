import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

# ==========================================================
# IMPORT CONFIG
# ==========================================================
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

import config


OUTPUT_DIR = Path(config.CNN_PROCESSED_DIR)
RESIZE_FOR_HOG = True  # Keep HOG feature size stable. If your eye crops are already 24x24, this does not change them.


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def load_features_from_split(split_csv: Path, use_ear: bool = True):
    df = pd.read_csv(split_csv)

    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        if RESIZE_FOR_HOG:
            img = cv2.resize(img, tuple(config.IMAGE_SIZE))

        img = img.astype(np.float32) / 255.0
        hog_feature = extract_hog_features(img)

        if use_ear:
            if "ear_avg" not in row or pd.isna(row["ear_avg"]):
                raise ValueError(f"Missing ear_avg for image: {image_path}")
            feature_vector = np.concatenate([hog_feature, [float(row["ear_avg"])]]).astype(np.float32)
        else:
            feature_vector = hog_feature.astype(np.float32)

        label = 0 if row["final_label"] == "open" else 1
        X.append(feature_vector)
        y.append(label)

    if skipped > 0:
        print(f"Warning: skipped {skipped} invalid images from {split_csv.name}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = OUTPUT_DIR / "train_split.csv"
    test_csv = OUTPUT_DIR / "test_split.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Run create_image_split.py first to create train_split.csv and test_split.csv.")

    print("===== SVM PREPROCESSING: HOG + EAR =====")
    print("Loading train split...")
    X_train, y_train = load_features_from_split(train_csv, use_ear=True)

    print("Loading test split...")
    X_test, y_test = load_features_from_split(test_csv, use_ear=True)

    print(f"Train before SMOTE: {X_train.shape}")
    print(f"Test             : {X_test.shape}")
    print("Train distribution before SMOTE:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test distribution             :", dict(zip(*np.unique(y_test, return_counts=True))))

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying SMOTE only on training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    np.save(OUTPUT_DIR / "X_train_img.npy", X_train_resampled)
    np.save(OUTPUT_DIR / "y_train_img.npy", y_train_resampled)
    np.save(OUTPUT_DIR / "X_test_img.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "y_test_img.npy", y_test)
    joblib.dump(scaler, OUTPUT_DIR / "scaler_img.pkl")

    print("\nDone.")
    print("Final train distribution after SMOTE:", dict(zip(*np.unique(y_train_resampled, return_counts=True))))
    print("Final test distribution:", dict(zip(*np.unique(y_test, return_counts=True))))
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
