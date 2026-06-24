import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

def dynamic_normalize(x):
    """
    Chuẩn hóa tỷ lệ theo giá trị Max tích lũy (Causal Max Normalization / Min=0).
    Khắc phục lỗi 'Khởi động lạnh' (Cold-start) do expanding min bám đuổi nhiễu mắt mở.
    """
    exp_max = x.expanding(min_periods=1).max()
    
    # Nếu exp_max = 0, trả về 0.5 để tránh lỗi chia cho 0
    norm = np.where(exp_max == 0, 0.5, x / exp_max)
    return pd.Series(norm, index=x.index)

def process_sequences(input_csv, output_dir, window_size=7):
    print(f"1. Đang tải dữ liệu từ {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
        
    df = df.drop_duplicates(subset=['video_id', 'frame_index']).copy()
    
    # Chuẩn hóa Min-Max EAR theo chiều thời gian thực (Causal Normalization)
    print("2. Đang chuẩn hóa Min-Max EAR (Causal/Dynamic) theo từng video...")
    # Cần sort theo time trước khi expanding
    df = df.sort_values(['video_id', 'frame_index'])
    df['ear_norm'] = df.groupby('video_id')['ear_avg'].transform(dynamic_normalize)
    
    print("3. Đang nội suy phục hồi các frame bị khuyết...")
    df_interp_list = []
    for video_id, group in df.groupby('video_id'):
        group = group.sort_values('frame_index').set_index('frame_index')
        full_idx = range(group.index.min(), group.index.max() + 1)
        group = group.reindex(full_idx)
        
        group['ear_avg'] = group['ear_avg'].interpolate(method='linear')
        group['ear_norm'] = group['ear_norm'].interpolate(method='linear')
        group['final_label'] = group['final_label'].ffill().bfill()
        group['video_id'] = video_id
        
        df_interp_list.append(group.reset_index())
        
    df_interp = pd.concat(df_interp_list, ignore_index=True)
    df_interp = df_interp.dropna(subset=['ear_avg'])
    
    # Tách tập Train/Test theo Video ID trước khi trích xuất cửa sổ
    # ĐỂ TRÁNH DATA LEAKAGE (Trùng lặp cửa sổ giữa Train và Test)
    print("\n4. Đang chia tập Train/Test theo Video (Chống Data Leakage)...")
    unique_videos = df_interp['video_id'].unique()
    train_vids, test_vids = train_test_split(unique_videos, test_size=0.2, random_state=42)
    
    print(f" - Tổng số Video: {len(unique_videos)}")
    print(f" - Tập Train    : {len(train_vids)} videos")
    print(f" - Tập Test     : {len(test_vids)} videos")
    
    # Chia DataFrame thành 2 phần rõ rệt
    df_train = df_interp[df_interp['video_id'].isin(train_vids)]
    df_test = df_interp[df_interp['video_id'].isin(test_vids)]
    
    def extract_windows(df_subset):
        X_sub, y_sub = [], []
        for video_id, group in df_subset.groupby('video_id'):
            group = group.sort_values('frame_index')
            ear_norm_values = group['ear_norm'].values
            label_values = group['final_label'].values
            
            if len(ear_norm_values) < window_size:
                continue
                
            for i in range(len(ear_norm_values) - window_size + 1):
                window_feat = ear_norm_values[i:i+window_size]
                window_labels = label_values[i:i+window_size]
                
                closed_frames = (window_labels == 'closed')
                closed_count = np.sum(closed_frames)
                
                if closed_count > 4:
                    label = 2
                elif 1 <= closed_count <= 4 and not closed_frames[0] and not closed_frames[-1]:
                    label = 1
                else:
                    label = 0
                    
                X_sub.append(window_feat)
                y_sub.append(label)
                
        if len(X_sub) == 0:
            return np.array([]), np.array([])
        return np.array(X_sub)[..., np.newaxis], np.array(y_sub)

    print("\n5. Đang trích xuất cửa sổ trượt (N=7)...")
    X_train, y_train = extract_windows(df_train)
    X_test, y_test = extract_windows(df_test)
    
    print("\n[BÁO CÁO TỔNG QUAN TẬP TRAIN]")
    print(f" - Số lượng cửa sổ (Windows): {len(X_train)}")
    print(f" - Nhãn 0 (No-Blink)    : {np.sum(y_train==0)}")
    print(f" - Nhãn 1 (Blink)       : {np.sum(y_train==1)}")
    print(f" - Nhãn 2 (Long-Closure): {np.sum(y_train==2)}")
    
    print("\n[BÁO CÁO TỔNG QUAN TẬP TEST]")
    print(f" - Số lượng cửa sổ (Windows): {len(X_test)}")
    print(f" - Nhãn 0 (No-Blink)    : {np.sum(y_test==0)}")
    print(f" - Nhãn 1 (Blink)       : {np.sum(y_test==1)}")
    print(f" - Nhãn 2 (Long-Closure): {np.sum(y_test==2)}")

    print("\n6. Đang lưu kết quả ra mảng Numpy...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_seq.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_seq.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test_seq.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test_seq.npy'), y_test)
    print(f"Hoàn thành! Các file npy đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiền xử lý chuỗi EAR (Sử dụng Ground-truth Labels & Chống Leakage)')
    parser.add_argument('--input', type=str, default=os.path.join('dataset_master', 'metadata_master.csv'), help='Đường dẫn file metadata gốc')
    parser.add_argument('--output', type=str, default=os.path.join('dataset_master', 'processed_seq'), help='Thư mục lưu X, y tập Train và Test')
    parser.add_argument('--window', type=int, default=7, help='Kích thước cửa sổ trượt (Mặc định: 7)')
    args = parser.parse_args()
    
    process_sequences(args.input, args.output, args.window)
