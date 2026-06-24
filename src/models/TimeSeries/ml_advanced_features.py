import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
    
    # Kurtosis (phi tuyến)
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

def run_ml_advanced_features():
    data_dir = os.path.join('dataset_master', 'processed_seq')
    X_train = np.load(os.path.join(data_dir, 'X_train_seq.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_seq.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_seq.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_seq.npy'))
    
    # Phẳng hóa thành 2D
    X_train_flat = X_train.squeeze(-1)
    X_test_flat = X_test.squeeze(-1)
    
    # Trích xuất đặc trưng 12 chiều không cộng tuyến
    X_train_adv = np.array([extract_advanced_features_single(w) for w in X_train_flat])
    X_test_adv = np.array([extract_advanced_features_single(w) for w in X_test_flat])
    
    # Định nghĩa các mô hình
    models = {
        'Linear SVM (C=0.1)': make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1, class_weight='balanced', random_state=42)),
        'Linear SVM (C=1.0)': make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)),
        'Linear SVM (C=5.0)': make_pipeline(StandardScaler(), SVC(kernel='linear', C=5.0, class_weight='balanced', random_state=42)),
        'RBF SVM (C=0.5)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=0.5, class_weight='balanced', random_state=42)),
        'RBF SVM (C=1.5)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.5, class_weight='balanced', random_state=42)),
        'RBF SVM (C=5.0)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=5.0, class_weight='balanced', random_state=42)),
        'Random Forest (Tree=100, Depth=6)': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42),
        'Random Forest (Tree=100, Depth=12)': RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42),
        'Gradient Boosting (Tree=100, LR=0.05)': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'Gradient Boosting (Tree=150, LR=0.10)': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        'KNN (K=3)': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        'KNN (K=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'KNN (K=7)': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        'Decision Tree (Depth=5)': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Decision Tree (Depth=8)': DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
        'Logistic Regression (balanced)': make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)),
        'Naive Bayes (Gaussian)': GaussianNB()
    }
    
    results = []
    
    print("\n--- Training and evaluating models on 12 advanced features (No Collinearity) ---")
    for name, clf in models.items():
        clf.fit(X_train_adv, y_train)
        
        start_time = time.time()
        y_pred = clf.predict(X_test_adv)
        inf_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        config_str = name.split('(')[-1].replace(')', '') if '(' in name else 'default'
        model_type = name.split('(')[0].strip()
        
        results.append({
            'model_name': f"ML Adv - {model_type}",
            'feature_type': 'advanced_12feature',
            'config': config_str,
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
        print(f" • Model: {name:<35} | Accuracy: {acc:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f}")
        
    return results, models

if __name__ == '__main__':
    run_ml_advanced_features()
