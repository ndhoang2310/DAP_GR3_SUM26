import sys
import joblib
import numpy as np
import random
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score)
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# IMPORT CONFIG
# ==========================================================

base_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_dir))

import config
from cnn_model import get_cnn_model


# ==========================================================
# CLASS WEIGHT
# ==========================================================

def get_class_weights(y_train):

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train)

    return dict(zip(classes, weights))


# ==========================================================
# TRAIN
# ==========================================================

def train_dl():

    print("\n===== CNN TRAINING =====")

    processed_dir = Path(config.CNN_PROCESSED_DIR)

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    X_train = np.load(processed_dir / "X_train_cnn.npy")

    y_train = np.load(processed_dir / "y_train_cnn.npy")

    X_val = np.load(processed_dir / "X_val_cnn.npy")

    y_val = np.load(processed_dir / "y_val_cnn.npy")

    X_test = np.load(processed_dir / "X_test_cnn.npy")

    y_test = np.load(processed_dir / "y_test_cnn.npy")

    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    # ------------------------------------------------------
    # CLASS WEIGHT
    # ------------------------------------------------------

    class_weights = get_class_weights(y_train)

    print("\nClass Weights:")
    print(class_weights)

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model = get_cnn_model(
        input_shape=(*config.IMAGE_SIZE, 1)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    Path(config.MODEL_SAVE_PATH).mkdir(
        parents=True,
        exist_ok=True)

    # ------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1)

    checkpoint = ModelCheckpoint(
        Path(config.MODEL_SAVE_PATH) / "best_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1)

    # ------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------

    print("\nTraining...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            early_stop,
            checkpoint],
        verbose=1)

    # ------------------------------------------------------
    # ACTUAL EPOCHS
    # ------------------------------------------------------

    actual_epochs = len(history.history["loss"])

    print(f"\nActual Epochs: {actual_epochs}")

    # ------------------------------------------------------
    # SAVE HISTORY
    # ------------------------------------------------------

    joblib.dump(history.history,processed_dir / "cnn_history.pkl")

    # ------------------------------------------------------
    # TEST
    # ------------------------------------------------------

    print("\n===== TEST EVALUATION =====")

    y_pred_prob = model.predict(X_test,verbose=0)

    y_pred = (y_pred_prob > 0.5).astype(int)

    y_pred = y_pred.flatten()

    # ------------------------------------------------------
    # ACCURACY
    # ------------------------------------------------------

    acc = accuracy_score(y_test,y_pred)

    print(f"Accuracy : {acc:.4f}")

    # ------------------------------------------------------
    # AUC
    # ------------------------------------------------------

    auc = roc_auc_score(y_test,y_pred_prob.ravel())

    print(f"AUC      : {auc:.4f}")

    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Open",
            "Closed"
        ],
        output_dict=True)

    print(f"Open F1   : {report['Open']['f1-score']:.4f}")

    print(f"Closed F1 : {report['Closed']['f1-score']:.4f}")

    print(f"Macro F1  : {report['macro avg']['f1-score']:.4f}")

    print(f"Weighted F1 : {report['weighted avg']['f1-score']:.4f}")

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    cm = confusion_matrix(y_test,y_pred)

    print("\nConfusion Matrix")

    print("               Pred Open  Pred Closed")

    print(f"True Open      {cm[0][0]}      {cm[0][1]}")

    print(f"True Closed    {cm[1][0]}      {cm[1][1]}")

    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------

    save_path = (Path(config.MODEL_SAVE_PATH)/ "cnn_blink_model.keras")

    model.save(save_path)

    print(f"\nModel saved at: {save_path}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    train_dl()