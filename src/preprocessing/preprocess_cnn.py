import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ==========================================================
# IMPORT CONFIG
# ==========================================================
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))

import config


OUTPUT_DIR = Path(config.CNN_PROCESSED_DIR)


def load_images_from_split(split_csv: Path):
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

        img = cv2.resize(img, tuple(config.IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        label = 0 if row["final_label"] == "open" else 1
        X.append(img)
        y.append(label)

    if skipped > 0:
        print(f"Warning: skipped {skipped} invalid images from {split_csv.name}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def save_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    np.save(OUTPUT_DIR / f"X_{name}_cnn.npy", X)
    np.save(OUTPUT_DIR / f"y_{name}_cnn.npy", y)
    print(f"{name:<5}: X={X.shape}, y={y.shape}, distribution={dict(zip(*np.unique(y, return_counts=True)))}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = OUTPUT_DIR / "train_split.csv"
    val_csv = OUTPUT_DIR / "val_split.csv"
    test_csv = OUTPUT_DIR / "test_split.csv"

    for path in [train_csv, val_csv, test_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}. Run create_image_split.py first.")

    print("===== CNN PREPROCESSING FROM SHARED SPLITS =====")
    X_train, y_train = load_images_from_split(train_csv)
    X_val, y_val = load_images_from_split(val_csv)
    X_test, y_test = load_images_from_split(test_csv)

    save_split("train", X_train, y_train)
    save_split("val", X_val, y_val)
    save_split("test", X_test, y_test)

    print("\nDone. Saved CNN arrays to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
