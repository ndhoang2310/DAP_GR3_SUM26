import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.models import load_model

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))

import config


def load_threshold(processed_dir: Path) -> float:
    threshold_path = processed_dir / "best_cnn_threshold.txt"
    if threshold_path.exists():
        return float(threshold_path.read_text(encoding="utf-8").strip())
    return 0.50


def summarize_model(name: str, y_true, y_pred, y_prob=None):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Open", "Closed"],
        output_dict=True,
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy    : {acc:.4f}")
    if auc is not None:
        print(f"AUC         : {auc:.4f}")
    print(f"Open F1     : {report['Open']['f1-score']:.4f}")
    print(f"Closed F1   : {report['Closed']['f1-score']:.4f}")
    print(f"Macro F1    : {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1 : {report['weighted avg']['f1-score']:.4f}")
    print("\nConfusion Matrix")
    print("               Pred Open  Pred Closed")
    print(f"True Open      {cm[0][0]}      {cm[0][1]}")
    print(f"True Closed    {cm[1][0]}      {cm[1][1]}")

    return {
        "name": name,
        "accuracy": acc,
        "auc": auc,
        "open_f1": report["Open"]["f1-score"],
        "closed_f1": report["Closed"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Open", "Closed"],
    )
    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    processed_dir = Path(config.CNN_PROCESSED_DIR)
    model_dir = Path(config.MODEL_SAVE_PATH)
    figures_dir = base_dir / "reports" / "figures"

    print("===== IMAGE MODEL BENCHMARK =====")

    # ======================================================
    # Load SVM HOG+EAR
    # ======================================================
    X_test_img = np.load(processed_dir / "X_test_img.npy")
    y_test_img = np.load(processed_dir / "y_test_img.npy")
    svm_model = joblib.load(model_dir / "svm_model.pkl")

    y_pred_svm = svm_model.predict(X_test_img)
    y_prob_svm = svm_model.predict_proba(X_test_img)[:, 1]
    svm_result = summarize_model("SVM HOG+EAR", y_test_img, y_pred_svm, y_prob_svm)
    save_confusion_matrix(
        svm_result["confusion_matrix"],
        "SVM HOG+EAR Confusion Matrix",
        figures_dir / "confusion_matrix_svm_hog_ear.png",
    )

    # ======================================================
    # Load SVM HOG-only if available
    # ======================================================
    hog_only_model_path = model_dir / "svm_hog_only_model.pkl"
    hog_only_x_path = processed_dir / "X_test_hog_only.npy"
    hog_only_y_path = processed_dir / "y_test_hog_only.npy"

    if hog_only_model_path.exists() and hog_only_x_path.exists() and hog_only_y_path.exists():
        X_test_hog_only = np.load(hog_only_x_path)
        y_test_hog_only = np.load(hog_only_y_path)
        hog_only_model = joblib.load(hog_only_model_path)
        y_pred_hog_only = hog_only_model.predict(X_test_hog_only)
        y_prob_hog_only = hog_only_model.predict_proba(X_test_hog_only)[:, 1]
        hog_only_result = summarize_model("SVM HOG-only", y_test_hog_only, y_pred_hog_only, y_prob_hog_only)
        save_confusion_matrix(
            hog_only_result["confusion_matrix"],
            "SVM HOG-only Confusion Matrix",
            figures_dir / "confusion_matrix_svm_hog_only.png",
        )

    # ======================================================
    # Load CNN
    # ======================================================
    X_test_cnn = np.load(processed_dir / "X_test_cnn.npy")
    y_test_cnn = np.load(processed_dir / "y_test_cnn.npy")

    cnn_model_path = model_dir / "best_cnn.keras"
    if not cnn_model_path.exists():
        cnn_model_path = model_dir / "cnn_blink_model.keras"

    cnn_model = load_model(cnn_model_path)
    threshold = load_threshold(processed_dir)
    y_prob_cnn = cnn_model.predict(X_test_cnn, verbose=0).ravel()
    y_pred_cnn = (y_prob_cnn > threshold).astype(int)

    cnn_result = summarize_model(f"CNN threshold={threshold:.2f}", y_test_cnn, y_pred_cnn, y_prob_cnn)
    save_confusion_matrix(
        cnn_result["confusion_matrix"],
        "CNN Confusion Matrix",
        figures_dir / "confusion_matrix_cnn.png",
    )

    print("\nSaved confusion matrices to:", figures_dir)

    # Extra check: labels should match if both pipelines use the same test_split.csv.
    if len(y_test_img) == len(y_test_cnn) and np.array_equal(y_test_img, y_test_cnn):
        print("\nShared test label check: PASSED. SVM and CNN test labels match.")
    else:
        print("\nShared test label check: WARNING. y_test_img and y_test_cnn do not match exactly.")
        print("This can happen if preprocessing skipped different images. Check image paths and split files.")


if __name__ == "__main__":
    main()
