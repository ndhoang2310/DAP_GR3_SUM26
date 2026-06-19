import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def process_sequences(input_csv, output_dir, window_size=7, ear_threshold=0.1050):
    print(f"1. Đang tải dữ liệu từ {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()
        
    # Loại bỏ các dòng trùng lặp về thời gian (nếu có)
    df = df.drop_duplicates(subset=['video_id', 'frame_index']).copy()
    
    # Tạo mã thành viên để chuẩn hóa cá nhân (ví dụ: M01_1 -> M01)
    df['member_id'] = df['video_id'].apply(lambda x: str(x).split('_')[0])
    
    # Chuẩn hóa Min-Max EAR per person (Dựa trên ear_avg)
    print("2. Đang chuẩn hóa Min-Max EAR theo từng thành viên...")
    df['ear_norm'] = df.groupby('member_id')['ear_avg'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )
    
    # Nội suy phục hồi (Linear Interpolation)
    print("3. Đang nội suy phục hồi các frame bị khuyết...")
    df_interp_list = []
    for video_id, group in df.groupby('video_id'):
        group = group.sort_values('frame_index').set_index('frame_index')
        # Tạo index liên tục từ frame nhỏ nhất đến lớn nhất
        full_idx = range(group.index.min(), group.index.max() + 1)
        group = group.reindex(full_idx)
        
        # Điền tuyến tính cho các frame bị mất
        group['ear_avg'] = group['ear_avg'].interpolate(method='linear')
        group['ear_norm'] = group['ear_norm'].interpolate(method='linear')
        group['video_id'] = video_id
        
        df_interp_list.append(group.reset_index())
        
    df_interp = pd.concat(df_interp_list, ignore_index=True)
    df_interp = df_interp.dropna(subset=['ear_avg']) # Xóa các NaN thừa ở đầu/cuối chuỗi nếu có
    
    # Cắt cửa sổ trượt và dán nhãn V-Shape
    print(f"4. Đang trích xuất cửa sổ trượt (N={window_size}) và dán nhãn...")
    X = []
    y = []
    
    for video_id, group in df_interp.groupby('video_id'):
        group = group.sort_values('frame_index')
        ear_values = group['ear_avg'].values # Dùng EAR gốc để phân biệt Mở/Nhắm theo Threshold
        ear_norm_values = group['ear_norm'].values # Dùng EAR chuẩn hóa làm đặc trưng (Feature)
        
        if len(ear_values) < window_size:
            continue
            
        for i in range(len(ear_values) - window_size + 1):
            window_ear = ear_values[i:i+window_size]
            window_feat = ear_norm_values[i:i+window_size]
            
            # LOGIC GÁN NHÃN V-SHAPE
            # Tìm các frame có giá trị EAR < Threshold (Đang nhắm mắt)
            closed_frames = window_ear < ear_threshold
            closed_count = np.sum(closed_frames)
            
            if closed_count > 4:
                label = 2 # Long-Closure (Ngủ gật / nhắm quá lâu)
            elif 2 <= closed_count <= 4 and not closed_frames[0] and not closed_frames[-1]:
                label = 1 # Blink (Nháy mắt chữ V hoàn hảo, hai biên mở)
            else:
                label = 0 # No-Blink / Transition (Bình thường hoặc chuyển trạng thái)
                
            X.append(window_feat)
            y.append(label)
            
    # Chuyển đổi sang mảng Numpy: shape (samples, timesteps, features)
    X = np.array(X)[..., np.newaxis] 
    y = np.array(y)
    
    print("\n[BÁO CÁO TỔNG QUAN]")
    print(f" - Tổng số cửa sổ trích xuất : {len(X)}")
    print(f" - Số lượng nhãn 0 (No-Blink)    : {np.sum(y==0)}")
    print(f" - Số lượng nhãn 1 (Blink)       : {np.sum(y==1)}")
    print(f" - Số lượng nhãn 2 (Long-Closure): {np.sum(y==2)}")
    
    # Chia tập Train/Test (80/20) với Stratify để đảm bảo tỷ lệ nhãn đồng đều
    print("\n5. Đang chia tập dữ liệu Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f" - Tập Train: {X_train.shape[0]} mẫu")
    print(f" - Tập Test : {X_test.shape[0]} mẫu")

    # Xuất file
    print("\n6. Đang lưu kết quả ra mảng Numpy...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_seq.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train_seq.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test_seq.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test_seq.npy'), y_test)
    print(f"Hoàn thành! Các file npy đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiền xử lý chuỗi EAR cho Mô hình Hướng 2')
    parser.add_argument('--input', type=str, default=os.path.join('dataset_master', 'metadata_master.csv'), help='Đường dẫn file metadata gốc')
    parser.add_argument('--output', type=str, default=os.path.join('dataset_master', 'processed_seq'), help='Thư mục lưu X, y tập Train và Test')
    parser.add_argument('--window', type=int, default=7, help='Kích thước cửa sổ trượt (Mặc định: 7)')
    parser.add_argument('--threshold', type=float, default=0.1050, help='Ngưỡng EAR nhắm mắt (Mặc định: 0.1050)')
    args = parser.parse_args()
    
    process_sequences(args.input, args.output, args.window, args.threshold)
