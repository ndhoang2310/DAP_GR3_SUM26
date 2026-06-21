import numpy as np
import joblib

from pathlib import Path

from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score)


def train_ml():

    # ======================================================
    # PATHS
    # ======================================================

    base_dir = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        .parent
        .parent)

    processed_dir = base_dir / "processed_image"
    model_dir = base_dir / "models"

    print(f"Đang load dữ liệu từ: {processed_dir}")

    # ======================================================
    # LOAD DATA
    # ======================================================

    X_train = np.load(processed_dir / "X_train_img.npy")

    y_train = np.load(processed_dir / "y_train_img.npy")

    X_test = np.load(processed_dir / "X_test_img.npy")

    y_test = np.load(processed_dir / "y_test_img.npy")

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    print("\n===== TRAINING SVM =====")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True)

    model.fit(
        X_train,
        y_train)

    # ======================================================
    # PREDICTION
    # ======================================================

    y_pred = model.predict(
        X_test
    )

    y_pred_prob = model.predict_proba(
        X_test
    )[:, 1]

    # ======================================================
    # ACCURACY
    # ======================================================

    acc = accuracy_score(
        y_test,
        y_pred)

    print("\n===== TEST EVALUATION =====")

    print(f"Accuracy : {acc:.4f}")

    # ======================================================
    # AUC
    # ======================================================

    auc = roc_auc_score(
        y_test,
        y_pred_prob)

    print(f"AUC      : {auc:.4f}")

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Open",
            "Closed"
        ],
        output_dict=True
    )

    print(f"Open F1      : {report['Open']['f1-score']:.4f}")

    print(f"Closed F1    : {report['Closed']['f1-score']:.4f}")

    print(f"Macro F1     : {report['macro avg']['f1-score']:.4f}")

    print(f"Weighted F1  : {report['weighted avg']['f1-score']:.4f}")

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix")

    print("               Pred Open  Pred Closed")

    print(f"True Open      {cm[0][0]}      {cm[0][1]}")

    print(f"True Closed    {cm[1][0]}      {cm[1][1]}")

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_dir.mkdir(
        exist_ok=True)

    model_path = (
        model_dir
        / "svm_model.pkl")

    joblib.dump(
        model,
        model_path)

    print(f"\nMô hình đã được lưu tại: {model_path}")


if __name__ == "__main__":
    train_ml()