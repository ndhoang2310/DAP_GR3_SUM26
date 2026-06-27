import os
import sys
# pyrefly: ignore [missing-import]
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
    Chuẩn hóa tỷ lệ lai (Hybrid Calibration):
    1. Khởi tạo mốc max bằng Max của 15 frame đầu tiên.
    2. Áp dụng Hard Lower Bound chống lỗi nhắm mắt khi bật camera.
    3. Từ frame 16 trở đi, tự động cập nhật baseline tăng lên nếu thấy EAR lớn hơn (expanding max).
    """
    x_vals = x.values
    init_max = np.max(x_vals[:15]) if len(x_vals) >= 15 else np.max(x_vals)
    
    # Áp dụng Hard Lower Bound chống khởi động lạnh lúc nhắm mắt
    if np.isnan(init_max) or init_max < 0.18:
        init_max = 0.23  # Ngưỡng mở mắt tối thiểu mặc định
        
    norm = np.zeros_like(x_vals)
    current_max = init_max
    for idx, val in enumerate(x_vals):
        if val > current_max:
            current_max = val
        norm[idx] = val / current_max
        
    return pd.Series(np.clip(norm, 0.0, 1.0), index=x.index)

def process_sequences(input_csv, output_dir, window_size=7):
    print(f"1. Đang tải dữ liệu từ {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
        
    df = df.drop_duplicates(subset=['video_id', 'frame_index']).copy()
    
    # 2. Nội suy phục hồi các frame bị khuyết trước khi chuẩn hóa
    print("2. Đang nội suy phục hồi các frame bị khuyết...")
    df_interp_list = []
    for video_id, group in df.groupby('video_id'):
        group = group.sort_values('frame_index').set_index('frame_index')
        full_idx = range(group.index.min(), group.index.max() + 1)
        group = group.reindex(full_idx)
        
        group['ear_avg'] = group['ear_avg'].interpolate(method='linear')
        group['final_label'] = group['final_label'].ffill().bfill()
        group['video_id'] = video_id
        
        df_interp_list.append(group.reset_index())
        
    df_interp = pd.concat(df_interp_list, ignore_index=True)
    df_interp = df_interp.dropna(subset=['ear_avg'])
    
    # 3. Chuẩn hóa EAR sử dụng thuật toán Hybrid (Median 15 frame đầu + Expanding Max sau đó)
    print("3. Đang chuẩn hóa EAR (Hybrid first_15_median + expanding_max)...")
    df_interp = df_interp.sort_values(['video_id', 'frame_index'])
    df_interp['ear_norm'] = df_interp.groupby('video_id')['ear_avg'].transform(dynamic_normalize)
    
    # 4. Tách tập Train/Val/Test theo Video ID (Chống Data Leakage)
    print("\n4. Đang chia tập Train/Val/Test theo Video (Chống Data Leakage)...")
    unique_videos = df_interp['video_id'].unique()
    train_vids, test_vids = train_test_split(unique_videos, test_size=0.2, random_state=42)
    train_vids_sub, val_vids = train_test_split(train_vids, test_size=0.15, random_state=42)
    
    print(f" - Tổng số Video    : {len(unique_videos)}")
    print(f" - Tập Train (sub)  : {len(train_vids_sub)} videos")
    print(f" - Tập Val (cho DL) : {len(val_vids)} videos")
    print(f" - Tập Test (độc lập): {len(test_vids)} videos")
    
    df_train = df_interp[df_interp['video_id'].isin(train_vids_sub)]
    df_val = df_interp[df_interp['video_id'].isin(val_vids)]
    df_test = df_interp[df_interp['video_id'].isin(test_vids)]
    
    def extract_windows(df_subset):
        X_sub, y_sub = [], []
        for video_id, group in df_subset.groupby('video_id'):
            group = group.sort_values('frame_index')
            ear_norm_values = group['ear_norm'].values
            label_values = group['final_label'].values
            
            # --- FIX: LOẠI BỎ 15 FRAME ĐẦU HIỆU CHUẨN KHỎI TẬP TRAIN/TEST ---
            # Chỉ bắt đầu trích xuất cửa sổ trượt từ frame thứ 16 trở đi (index 15)
            if len(ear_norm_values) < 15 + window_size:
                continue
                
            for i in range(15, len(ear_norm_values) - window_size + 1):
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

    print("\n5. Đang trích xuất cửa sổ trượt (N=7) bắt đầu từ frame 16...")
    X_train, y_train = extract_windows(df_train)
    X_val, y_val = extract_windows(df_val)
    X_test, y_test = extract_windows(df_test)
    
    print("\n[BÁO CÁO TỔNG QUAN TẬP TRAIN]")
    print(f" - Số lượng cửa sổ (Windows): {len(X_train)}")
    print(f" - Nhãn 0 (No-Blink)    : {np.sum(y_train==0)}")
    print(f" - Nhãn 1 (Blink)       : {np.sum(y_train==1)}")
    print(f" - Nhãn 2 (Long-Closure): {np.sum(y_train==2)}")
    
    print("\n[BÁO CÁO TỔNG QUAN TẬP VALIDATION]")
    print(f" - Số lượng cửa sổ (Windows): {len(X_val)}")
    print(f" - Nhãn 0 (No-Blink)    : {np.sum(y_val==0)}")
    print(f" - Nhãn 1 (Blink)       : {np.sum(y_val==1)}")
    print(f" - Nhãn 2 (Long-Closure): {np.sum(y_val==2)}")
    
    print("\n[BÁO CÁO TỔNG QUAN TẬP TEST]")
    print(f" - Số lượng cửa sổ (Windows): {len(X_test)}")
    print(f" - Nhãn 0 (No-Blink)    : {np.sum(y_test==0)}")
    print(f" - Nhãn 1 (Blink)       : {np.sum(y_test==1)}")
    print(f" - Nhãn 2 (Long-Closure): {np.sum(y_test==2)}")

    print("\n6. Đang lưu kết quả ra mảng Numpy...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_seq.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_seq.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val_seq.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val_seq.npy'), y_val)
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
