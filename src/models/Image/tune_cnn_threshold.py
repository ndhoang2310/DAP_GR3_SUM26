import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))

import config


def main():
    processed_dir = Path(config.CNN_PROCESSED_DIR)
    model_path = Path(config.MODEL_SAVE_PATH) / "best_cnn.keras"

    if not model_path.exists():
        model_path = Path(config.MODEL_SAVE_PATH) / "cnn_blink_model.keras"

    print("===== TUNE CNN THRESHOLD ON VALIDATION SET =====")
    print("Loading model:", model_path)

    model = load_model(model_path)
    X_val = np.load(processed_dir / "X_val_cnn.npy")
    y_val = np.load(processed_dir / "y_val_cnn.npy")

    y_prob = model.predict(X_val, verbose=0).ravel()

    best_threshold = 0.5
    best_macro_f1 = -1.0
    rows = []

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_prob > threshold).astype(int)

        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        closed_f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        closed_precision = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        closed_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

        rows.append((threshold, macro_f1, closed_f1, closed_precision, closed_recall))

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold

    print("\nThreshold  MacroF1  ClosedF1  ClosedPrecision  ClosedRecall")
    for threshold, macro_f1, closed_f1, closed_precision, closed_recall in rows:
        marker = " <-- best" if abs(threshold - best_threshold) < 1e-9 else ""
        print(f"{threshold:9.2f}  {macro_f1:7.4f}  {closed_f1:8.4f}  {closed_precision:15.4f}  {closed_recall:12.4f}{marker}")

    output_path = processed_dir / "best_cnn_threshold.txt"
    output_path.write_text(f"{best_threshold:.2f}\n", encoding="utf-8")

    print(f"\nBest threshold by Macro F1: {best_threshold:.2f}")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
