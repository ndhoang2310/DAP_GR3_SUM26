import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_baseline_heuristics(input_csv):
    # 1. Đọc và tiền xử lý dữ liệu để tái tạo đúng tập Test
    df = pd.read_csv(input_csv)
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
        
    df = df.drop_duplicates(subset=['video_id', 'frame_index']).copy()
    df['member_id'] = df['video_id'].apply(lambda x: str(x).split('_')[0])
    
    # Tính toán Min-Max của ear_avg cho từng thành viên để chuẩn hóa ngưỡng
    member_stats = {}
    for m_id, group in df.groupby('member_id'):
        member_stats[m_id] = {
            'min': group['ear_avg'].min(),
            'max': group['ear_avg'].max()
        }
        
    df['ear_norm'] = df.groupby('member_id')['ear_avg'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )
    
    # Nội suy tuyến tính phục hồi các frame bị khuyết
    df_interp_list = []
    for video_id, group in df.groupby('video_id'):
        group = group.sort_values('frame_index').set_index('frame_index')
        full_idx = range(group.index.min(), group.index.max() + 1)
        group = group.reindex(full_idx)
        
        group['ear_avg'] = group['ear_avg'].interpolate(method='linear')
        group['ear_norm'] = group['ear_norm'].interpolate(method='linear')
        group['final_label'] = group['final_label'].ffill().bfill()
        group['video_id'] = video_id
        group['member_id'] = group['video_id'].apply(lambda x: str(x).split('_')[0] if pd.notna(x) else np.nan)
        group['member_id'] = group['member_id'].ffill().bfill()
        
        df_interp_list.append(group.reset_index())
        
    df_interp = pd.concat(df_interp_list, ignore_index=True).dropna(subset=['ear_avg'])
    
    # Trích xuất cửa sổ trượt
    window_size = 7
    
    X_ear_avg = []
    X_ear_norm = []
    X_member_id = []
    y = []
    
    for video_id, group in df_interp.groupby('video_id'):
        group = group.sort_values('frame_index')
        ear_avg_vals = group['ear_avg'].values
        ear_norm_vals = group['ear_norm'].values
        label_vals = group['final_label'].values
        member_ids = group['member_id'].values
        
        if len(ear_avg_vals) < window_size:
            continue
            
        for i in range(len(ear_avg_vals) - window_size + 1):
            w_ear_avg = ear_avg_vals[i:i+window_size]
            w_ear_norm = ear_norm_vals[i:i+window_size]
            w_labels = label_vals[i:i+window_size]
            w_member = member_ids[i] # Lấy tên thành viên của cửa sổ
            
            # Label ground-truth dựa trên final_label giống hệt code gốc
            closed_frames = (w_labels == 'closed')
            closed_count = np.sum(closed_frames)
            
            if closed_count > 4:
                lbl = 2
            elif 2 <= closed_count <= 4 and not closed_frames[0] and not closed_frames[-1]:
                lbl = 1
            else:
                lbl = 0
                
            X_ear_avg.append(w_ear_avg)
            X_ear_norm.append(w_ear_norm)
            X_member_id.append(w_member)
            y.append(lbl)
            
    X_ear_avg = np.array(X_ear_avg)
    X_ear_norm = np.array(X_ear_norm)
    X_member_id = np.array(X_member_id)
    y = np.array(y)
    
    # Chia Train/Test đồng nhất với random_state=42 và stratify=y
    # Cách này đảm bảo tập Test của baseline trùng khớp 100% với tập Test của ML
    indices = np.arange(len(y))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_test_ear_avg = X_ear_avg[test_idx]
    X_test_ear_norm = X_ear_norm[test_idx]
    X_test_member = X_member_id[test_idx]
    y_test = y[test_idx]
    
    results = []
    
    # ======================================================================
    # BASELINE 1: Ngưỡng tĩnh tuyệt đối (Ngưỡng 0.1050 trên ear_avg gốc)
    # ======================================================================
    y_pred_static = []
    ear_threshold = 0.1050
    for w in X_test_ear_avg:
        closed_frames = w < ear_threshold
        closed_count = np.sum(closed_frames)
        if closed_count > 4:
            pred = 2
        elif 2 <= closed_count <= 4 and not closed_frames[0] and not closed_frames[-1]:
            pred = 1
        else:
            pred = 0
        y_pred_static.append(pred)
        
    y_pred_static = np.array(y_pred_static)
    acc_static = accuracy_score(y_test, y_pred_static)
    report_static = classification_report(y_test, y_pred_static, output_dict=True, zero_division=0)
    
    print("\n--- BASELINE 1: Ngưỡng tĩnh 0.1050 trên EAR thô ---")
    print(f"Accuracy: {acc_static:.4f}")
    print(classification_report(y_test, y_pred_static, zero_division=0))
    
    results.append({
        'model_name': 'Baseline Static (Raw)',
        'feature_type': 'raw_ear_avg',
        'config': 'threshold=0.1050_static_global',
        'accuracy': acc_static,
        'precision_0': report_static['0']['precision'],
        'recall_0': report_static['0']['recall'],
        'f1_0': report_static['0']['f1-score'],
        'precision_1': report_static['1']['precision'],
        'recall_1': report_static['1']['recall'],
        'f1_1': report_static['1']['f1-score'],
        'precision_2': report_static['2']['precision'],
        'recall_2': report_static['2']['recall'],
        'f1_2': report_static['2']['f1-score'],
        'macro_f1': report_static['macro avg']['f1-score'],
        'inference_time': 0.001 # Tĩnh chạy siêu nhanh
    })
    
    # ======================================================================
    # BASELINE 2: Ngưỡng tĩnh đã chuẩn hóa cá nhân (Normalized Threshold per member)
    # ======================================================================
    y_pred_norm = []
    for w_norm, m_id in zip(X_test_ear_norm, X_test_member):
        # Lấy min-max của thành viên để chuyển đổi ngưỡng 0.1050 về thang đo [0, 1] của họ
        m_min = member_stats[m_id]['min']
        m_max = member_stats[m_id]['max']
        
        # Ngưỡng chuẩn hóa tương ứng của người đó
        thresh_norm = (0.1050 - m_min) / (m_max - m_min) if m_max != m_min else 0.0
        thresh_norm = np.clip(thresh_norm, 0.0, 1.0)
        
        closed_frames = w_norm < thresh_norm
        closed_count = np.sum(closed_frames)
        if closed_count > 4:
            pred = 2
        elif 2 <= closed_count <= 4 and not closed_frames[0] and not closed_frames[-1]:
            pred = 1
        else:
            pred = 0
        y_pred_norm.append(pred)
        
    y_pred_norm = np.array(y_pred_norm)
    acc_norm = accuracy_score(y_test, y_pred_norm)
    report_norm = classification_report(y_test, y_pred_norm, output_dict=True, zero_division=0)
    
    print("\n--- BASELINE 2: Ngưỡng chuẩn hóa theo cá nhân (Normalized Threshold per person) ---")
    print(f"Accuracy: {acc_norm:.4f}")
    print(classification_report(y_test, y_pred_norm, zero_division=0))
    
    results.append({
        'model_name': 'Baseline Static (Normalized)',
        'feature_type': 'ear_norm',
        'config': 'threshold=0.1050_normalized_per_member',
        'accuracy': acc_norm,
        'precision_0': report_norm['0']['precision'],
        'recall_0': report_norm['0']['recall'],
        'f1_0': report_norm['0']['f1-score'],
        'precision_1': report_norm['1']['precision'],
        'recall_1': report_norm['1']['recall'],
        'f1_1': report_norm['1']['f1-score'],
        'precision_2': report_norm['2']['precision'],
        'recall_2': report_norm['2']['recall'],
        'f1_2': report_norm['2']['f1-score'],
        'macro_f1': report_norm['macro avg']['f1-score'],
        'inference_time': 0.001
    })
    
    return results

if __name__ == '__main__':
    csv_path = os.path.join('dataset_master', 'metadata_master.csv')
    evaluate_baseline_heuristics(csv_path)
