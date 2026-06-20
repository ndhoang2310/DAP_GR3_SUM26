import os
import sys
import pandas as pd

# Thêm thư mục hiện tại vào sys.path để import các module local bất kể CWD
ts_dir = os.path.dirname(os.path.abspath(__file__))
if ts_dir not in sys.path:
    sys.path.append(ts_dir)

from baseline_static import evaluate_baseline_heuristics
from ml_raw_7frame import run_ml_raw_7frame
from ml_advanced_features import run_ml_advanced_features
from hybrid_system import run_hybrid_system
from lstm_model import run_lstm_training

def run_all_benchmarks():
    print("=" * 70)
    print("   KHỞI CHẠY HỆ THỐNG ĐÁNH GIÁ TOÀN DIỆN HƯỚNG 2 (TIMESERIES EAR)")
    print("=" * 70)
    
    # 1. Chạy Baseline tĩnh
    csv_path = os.path.join('dataset_master', 'metadata_master.csv')
    baseline_results = evaluate_baseline_heuristics(csv_path)
    
    # 2. Chạy ML trên 7 đặc trưng EAR thô
    ml_raw_results = run_ml_raw_7frame()
    
    # 3. Chạy ML trên 26 đặc trưng cải tiến
    ml_adv_results = run_ml_advanced_features()
    
    # 4. Chạy mô hình Deep Learning (LSTM - Option B)
    lstm_results = run_lstm_training()
    
    # 5. Chạy hệ thống Hybrid
    hybrid_results = run_hybrid_system()
    
    # Gộp tất cả kết quả
    all_results = baseline_results + ml_raw_results + ml_adv_results + lstm_results + hybrid_results
    
    # Tạo DataFrame
    df = pd.DataFrame(all_results)
    
    # Sắp xếp kết quả theo Macro F1 để dễ so sánh hiệu năng
    df_sorted = df.sort_values(by='macro_f1', ascending=False)
    
    # In bảng so sánh đẹp mắt ra console
    print("\n" + "=" * 100)
    print("                           BẢNG XẾP HẠNG BENCHMARK TOÀN BỘ HƯỚNG 2")
    print("=" * 100)
    print(f"{'Mô hình':<25} | {'Đặc trưng':<20} | {'Cấu hình / Config':<45} | {'Acc':<6} | {'Macro F1':<8} | {'Time':<6}")
    print("-" * 115)
    for idx, row in df_sorted.iterrows():
        print(f"{row['model_name']:<25} | {row['feature_type']:<20} | {row['config']:<45} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['inference_time']:.4f}s")
    print("=" * 115)
    
    # Lưu kết quả vào file CSV trong dataset_master kèm thông số config
    output_csv = os.path.join('dataset_master', 'benchmark_results.csv')
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n[THÀNH CÔNG] Đã lưu bảng kết quả benchmark chi tiết kèm cấu hình tại: {output_csv}")

if __name__ == '__main__':
    run_all_benchmarks()
