import joblib
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.svm import SVC


def print_evaluation(y_test, y_pred, y_pred_prob):
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Open", "Closed"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    print("\n===== TEST EVALUATION =====")
    print(f"Accuracy    : {acc:.4f}")
    print(f"AUC         : {auc:.4f}")
    print(f"Open F1     : {report['Open']['f1-score']:.4f}")
    print(f"Closed F1   : {report['Closed']['f1-score']:.4f}")
    print(f"Macro F1    : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1 : {report['weighted avg']['f1-score']:.4f}")

    print("\nConfusion Matrix")
    print("               Pred Open  Pred Closed")
    print(f"True Open      {cm[0][0]}      {cm[0][1]}")
    print(f"True Closed    {cm[1][0]}      {cm[1][1]}")


def train_ml_hog_only():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    processed_dir = base_dir / "processed_image"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("===== TRAINING SVM: HOG ONLY =====")
    print(f"Loading data from: {processed_dir}")

    X_train = np.load(processed_dir / "X_train_hog_only.npy")
    y_train = np.load(processed_dir / "y_train_hog_only.npy")
    X_test = np.load(processed_dir / "X_test_hog_only.npy")
    y_test = np.load(processed_dir / "y_test_hog_only.npy")

    print(f"Train: {X_train.shape}, distribution={dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test : {X_test.shape}, distribution={dict(zip(*np.unique(y_test, return_counts=True)))}")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print_evaluation(y_test, y_pred, y_pred_prob)

    model_path = model_dir / "svm_hog_only_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved at: {model_path}")


if __name__ == "__main__":
    train_ml_hog_only()
