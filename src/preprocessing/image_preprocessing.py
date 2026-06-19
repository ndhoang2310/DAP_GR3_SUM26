import os
import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


DATASET_DIR = "dataset_master"
CSV_PATH = os.path.join(DATASET_DIR, "metadata_master.csv")

OUTPUT_DIR = "processed_image"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_hog_features(img):

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        feature_vector=True
    )


def load_dataset():

    df = pd.read_csv(CSV_PATH)

    df = df[
        (df["status"] == "success")
        &
        (df["final_label"].isin(["open", "closed"]))
    ].copy()

    X = []
    y = []

    for _, row in df.iterrows():

        image_path = row["image_path"]

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = img.astype(np.float32) / 255.0

        hog_feature = extract_hog_features(img)

        ear_avg = float(row["ear_avg"])

        feature_vector = np.concatenate(
            [hog_feature, [ear_avg]]
        )

        X.append(feature_vector)

        label = 0 if row["final_label"] == "open" else 1
        y.append(label)

    return np.array(X), np.array(y)


def save_numpy(X_train, y_train, X_test, y_test):

    np.save(
        os.path.join(OUTPUT_DIR, "X_train_img.npy"),
        X_train
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_train_img.npy"),
        y_train
    )

    np.save(
        os.path.join(OUTPUT_DIR, "X_test_img.npy"),
        X_test
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y_test_img.npy"),
        y_test
    )


def main():

    print("Loading dataset...")

    X, y = load_dataset()

    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    save_numpy(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("Done.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()