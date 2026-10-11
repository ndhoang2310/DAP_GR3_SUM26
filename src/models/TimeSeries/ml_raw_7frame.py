import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

def run_ml_raw_7frame():
    data_dir = os.path.join('dataset_master', 'processed_seq')
    X_train_sub = np.load(os.path.join(data_dir, 'X_train_seq.npy'))
    y_train_sub = np.load(os.path.join(data_dir, 'y_train_seq.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val_seq.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val_seq.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_seq.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_seq.npy'))
    
    # Concatenate train_sub and val to train on the full training set (20 videos)
    X_train = np.concatenate([X_train_sub, X_val], axis=0)
    y_train = np.concatenate([y_train_sub, y_val], axis=0)
    
    # Phẳng hóa thành 2D (samples, 7)
    X_train_flat = X_train.squeeze(-1)
    X_test_flat = X_test.squeeze(-1)
    
    # Định nghĩa các mô hình truyền thống với nhiều cấu hình Hyperparameter khác nhau
    models = {
        'Linear SVM (C=0.1)': make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1, class_weight='balanced', random_state=42)),
        'Linear SVM (C=1.0)': make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)),
        'Linear SVM (C=5.0)': make_pipeline(StandardScaler(), SVC(kernel='linear', C=5.0, class_weight='balanced', random_state=42)),
        'RBF SVM (C=0.5)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=0.5, class_weight='balanced', random_state=42)),
        'RBF SVM (C=1.5)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.5, class_weight='balanced', random_state=42)),
        'RBF SVM (C=5.0)': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=5.0, class_weight='balanced', random_state=42)),
        'Random Forest (Tree=100, Depth=6)': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42),
        'Random Forest (Tree=200, Depth=10)': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
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
    
    print("\n--- Training and evaluating models on 7 raw EAR features ---")
    for name, clf in models.items():
        # Huấn luyện mô hình
        clf.fit(X_train_flat, y_train)
        
        # Đo thời gian dự đoán trên tập test (đại diện cho độ trễ thời gian thực)
        start_time = time.time()
        y_pred = clf.predict(X_test_flat)
        inf_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Trích xuất config tham số từ tên mô hình
        config_str = name.split('(')[-1].replace(')', '') if '(' in name else 'default'
        model_type = name.split('(')[0].strip()
        
        results.append({
            'model_name': f"ML Raw - {model_type}",
            'feature_type': 'raw_7frame_ear',
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
    run_ml_raw_7frame()
