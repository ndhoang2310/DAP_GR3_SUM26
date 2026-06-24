import os
import time
import numpy as np

# Thử import tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import classification_report, accuracy_score
    HAS_TF = True
except ImportError:
    HAS_TF = False

def extract_3d_sequence_features(X_seq):
    samples = X_seq.shape[0]
    time_steps = X_seq.shape[1]
    X_new = np.zeros((samples, time_steps, 3))
    
    for i in range(samples):
        w = X_seq[i, :, 0]
        # 1. EAR_norm
        X_new[i, :, 0] = w
        # 2. Delta_EAR
        diff1 = np.diff(w)
        X_new[i, 1:, 1] = diff1
        X_new[i, 0, 1] = 0.0
        # 3. Delta_Delta_EAR
        diff2 = np.diff(diff1)
        X_new[i, 2:, 2] = diff2
        X_new[i, 0, 2] = 0.0
        X_new[i, 1, 2] = 0.0
        
    return X_new

def run_lstm_training():
    if not HAS_TF:
        print("\n[WARNING] 'tensorflow' is not installed. Skipping LSTM training in benchmark.")
        return [], None
        
    print("\n[INFO] Loading EAR sequences for LSTM (Option B - 3D Features)...")
    data_dir = os.path.join('dataset_master', 'processed_seq')
    X_train = np.load(os.path.join(data_dir, 'X_train_seq.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_seq.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_seq.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_seq.npy'))
    
    # Extract 3D sequence features (samples, 7, 3)
    X_train_3d = extract_3d_sequence_features(X_train)
    X_test_3d = extract_3d_sequence_features(X_test)
    
    # Define LSTM model
    model = Sequential([
        Input(shape=(7, 3)),
        LSTM(16, return_sequences=False),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("[INFO] Training LSTM model...")
    model.fit(
        X_train_3d, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0 # Disable detailed logs to keep console clean
    )
    
    # Inference and timing
    start_time = time.time()
    y_pred_probs = model.predict(X_test_3d, verbose=0)
    inf_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Save model
    save_dir = os.path.join('dataset_master', 'models')
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'lstm_blink_model.h5'))
    
    print(f"[SUCCESS] Done! LSTM Accuracy: {acc:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f}")
    
    return [{
        'model_name': 'DL - LSTM (Option B)',
        'feature_type': '3d_temporal_dynamic',
        'config': 'LSTM_units=16_dropout=0.5_epochs=40',
        'accuracy': acc,
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_1': report['1']['f1-score'],
        'precision_2': report['2']['precision'],
        'recall_2': report['2']['recall'],
        'f1_2': report['2']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'inference_time': inf_time
    }], model

if __name__ == '__main__':
    run_lstm_training()
