import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_advanced_features_single(w):
    min_val = np.min(w)
    max_val = np.max(w)
    mean_val = np.mean(w)
    std_val = np.std(w)
    rng_val = max_val - min_val
    
    drop_left = w[0] - w[3]
    drop_right = w[6] - w[3]
    ratio_center = (w[0] + w[6]) / (2 * w[3] + 1e-5)
    
    diff1 = np.diff(w)
    diff2 = np.diff(diff1)
    
    feat = np.concatenate([
        w,
        [min_val, max_val, mean_val, std_val, rng_val, drop_left, drop_right, ratio_center],
        diff1,
        diff2
    ])
    return feat

def run_hybrid_system():
    data_dir = os.path.join('dataset_master', 'processed_seq')
    X_train = np.load(os.path.join(data_dir, 'X_train_seq.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_seq.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_seq.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_seq.npy'))
    
    X_train_flat = X_train.squeeze(-1)
    X_test_flat = X_test.squeeze(-1)
    
    # 1. Trích xuất đặc trưng nâng cao cho tập Train để huấn luyện RF
    X_train_adv = np.array([extract_advanced_features_single(w) for w in X_train_flat])
    
    print("\n--- Đang huấn luyện mô hình Random Forest cho hệ thống Hybrid ---")
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(X_train_adv, y_train)
    
    # 2. Định nghĩa các ngưỡng lọc Heuristic t khác nhau
    thresholds = [0.2, 0.3, 0.35, 0.4, 0.5]
    results = []
    
    print("\n--- Đang đánh giá hệ thống Hybrid ---")
    for t in thresholds:
        start_time = time.time()
        y_pred = []
        ml_runs = 0
        
        for w in X_test_flat:
            # Kiểm tra xem có frame nào trong window có giá trị EAR chuẩn hóa < t không
            is_suspicious = np.min(w) < t
            
            if is_suspicious:
                feat = extract_advanced_features_single(w)
                pred = rf.predict([feat])[0]
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
            'model_name': f"Hybrid (Filter + RF)",
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
        print(f" • Ngưỡng t: {t:<4} | Tiết kiệm CPU: {skip_percent:.1f}% | Accuracy: {acc:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f}")
        
    return results

if __name__ == '__main__':
    run_hybrid_system()
