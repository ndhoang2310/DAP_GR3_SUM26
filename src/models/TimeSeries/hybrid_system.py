import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_advanced_features_single(w):
    """
    Trích xuất 12 đặc trưng động học độc lập (loại bỏ hoàn toàn collinearity):
    - 7 giá trị EAR thô trong window
    - min_val (phi tuyến)
    - max_val (phi tuyến)
    - std_val (phi tuyến)
    - ratio_center (tỷ lệ - phi tuyến)
    - kurtosis (độ nhọn - phi tuyến)
    """
    min_val = np.min(w)
    max_val = np.max(w)
    std_val = np.std(w)
    ratio_center = (w[0] + w[6]) / (2 * w[3] + 1e-5)
    
    # Kurtosis
    mean_val = np.mean(w)
    if std_val > 1e-5:
        kurtosis = np.mean(((w - mean_val) / std_val) ** 4) - 3.0
    else:
        kurtosis = 0.0
        
    feat = np.concatenate([
        w,
        [min_val, max_val, std_val, ratio_center, kurtosis]
    ])
    return feat

def run_hybrid_system(best_model=None, feature_type='advanced_12feature'):
    data_dir = os.path.join('dataset_master', 'processed_seq')
    X_train = np.load(os.path.join(data_dir, 'X_train_seq.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_seq.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_seq.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_seq.npy'))
    
    X_test_flat = X_test.squeeze(-1)
    
    # 1. Khởi tạo model chạy chính
    if best_model is None:
        X_train_sub = np.load(os.path.join(data_dir, 'X_train_seq.npy'))
        y_train_sub = np.load(os.path.join(data_dir, 'y_train_seq.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val_seq.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val_seq.npy'))
        X_train = np.concatenate([X_train_sub, X_val], axis=0)
        y_train = np.concatenate([y_train_sub, y_val], axis=0)
        
        X_train_flat = X_train.squeeze(-1)
        X_train_adv = np.array([extract_advanced_features_single(w) for w in X_train_flat])
        print("\n--- Training Random Forest model for Hybrid system (Fallback) ---")
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42)
        rf.fit(X_train_adv, y_train)
        best_model = rf
        feature_type = 'advanced_12feature'
        model_name_tag = "RF"
    else:
        underlying_model = best_model.steps[-1][1] if hasattr(best_model, 'steps') else best_model
        model_name_tag = type(underlying_model).__name__.replace("Classifier", "")
        print(f"\n--- Running Hybrid System with best model: {model_name_tag} using {feature_type} ---")
    
    # 2. Define different heuristic filter thresholds t
    thresholds = [0.2, 0.3, 0.35, 0.4, 0.5]
    results = []
    
    print("\n--- Evaluating Hybrid system ---")
    for t in thresholds:
        start_time = time.time()
        y_pred = []
        ml_runs = 0
        
        for w in X_test_flat:
            # Kiểm tra xem có frame nào trong window có giá trị EAR chuẩn hóa < t không
            is_suspicious = np.min(w) < t
            
            if is_suspicious:
                if feature_type == 'advanced_12feature':
                    feat = extract_advanced_features_single(w)
                else:
                    feat = w
                pred = best_model.predict([feat])[0]
                y_pred.append(pred)
                ml_runs += 1
            else:
                y_pred.append(0) # Trả về 0 ngay lập tức, bỏ qua ML
                
        inf_time = time.time() - start_time
        y_pred = np.array(y_pred)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        skip_percent = (1 - (ml_runs / len(X_test_flat))) * 100
        
        results.append({
            'model_name': f"Hybrid (Filter + {model_name_tag})",
            'feature_type': 'hybrid_ear_norm',
            'config': f"threshold={t}_skip_ml={skip_percent:.1f}%",
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
        })
        print(f" • Threshold t: {t:<4} | CPU Saved: {skip_percent:.1f}% | Accuracy: {acc:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f}")
        
    return results

if __name__ == '__main__':
    run_hybrid_system()
