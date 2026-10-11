import os
import sys
import pandas as pd

# Thêm thư mục hiện tại vào sys.path để import các module local bất kể CWD
ts_dir = os.path.dirname(os.path.abspath(__file__))
if ts_dir not in sys.path:
    sys.path.append(ts_dir)

import joblib
from ml_raw_7frame import run_ml_raw_7frame
from ml_advanced_features import run_ml_advanced_features
from hybrid_system import run_hybrid_system
from lstm_model import run_lstm_training

def run_all_benchmarks():
    print("=" * 70)
    print("   STARTING COMPREHENSIVE BENCHMARK PIPELINE - DIRECTION 2 (TimeSeries EAR)")
    print("=" * 70)
      
    # 2. Chạy ML trên 7 đặc trưng EAR thô
    ml_raw_results, ml_raw_models = run_ml_raw_7frame()
    
    # 3. Chạy ML trên 12 đặc trưng cải tiến (không cộng tuyến)
    ml_adv_results, ml_adv_models = run_ml_advanced_features()
    
    # 4. Chạy mô hình Deep Learning (LSTM - Option B)
    lstm_results, lstm_model = run_lstm_training()
    
    # Tìm mô hình ML truyền thống tốt nhất để chạy Hybrid
    all_ml_results = ml_raw_results + ml_adv_results
    df_ml = pd.DataFrame(all_ml_results)
    df_ml_sorted = df_ml.sort_values(by='macro_f1', ascending=False)
    
    best_row = df_ml_sorted.iloc[0]
    best_model_name = best_row['model_name']
    best_feature_type = best_row['feature_type']
    best_config = best_row['config']
    
    # Tìm đối tượng model thực tế
    best_model_obj = None
    if best_feature_type == 'advanced_12feature':
        model_type_key = best_model_name.replace("ML Adv - ", "")
        key = f"{model_type_key} ({best_config})" if best_config != 'default' else model_type_key
        best_model_obj = ml_adv_models[key]
    elif best_feature_type == 'raw_7frame_ear':
        model_type_key = best_model_name.replace("ML Raw - ", "")
        key = f"{model_type_key} ({best_config})" if best_config != 'default' else model_type_key
        best_model_obj = ml_raw_models[key]
        
    print(f"\n[INFO] Selected best ML model for Hybrid System: {best_model_name} ({best_config})")
    
    # 5. Chạy hệ thống Hybrid với mô hình tốt nhất
    hybrid_results = run_hybrid_system(best_model=best_model_obj, feature_type=best_feature_type)
    
    # Gộp tất cả kết quả
    all_results = ml_raw_results + ml_adv_results + lstm_results + hybrid_results
    
    # Tạo DataFrame
    df = pd.DataFrame(all_results)
    
    # Sắp xếp kết quả theo Macro F1 để dễ so sánh hiệu năng
    df_sorted = df.sort_values(by='macro_f1', ascending=False)
    
    # Print results leaderboard
    print("\n" + "=" * 115)
    print("                                     BENCHMARK RESULTS LEADERBOARD")
    print("=" * 115)
    print(f"{'Model Name':<25} | {'Feature Type':<20} | {'Hyperparameters / Config':<45} | {'Acc':<6} | {'Macro F1':<8} | {'Latency':<6}")
    print("-" * 115)
    for idx, row in df_sorted.iterrows():
        print(f"{row['model_name']:<25} | {row['feature_type']:<20} | {row['config']:<45} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['inference_time']:.4f}s")
    print("=" * 115)
    
    # Save results to CSV
    output_csv = os.path.join('dataset_master', 'benchmark_results.csv')
    try:
        df_sorted.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Saved benchmark results CSV to: {output_csv}")
    except PermissionError:
        backup_csv = os.path.join('dataset_master', 'benchmark_results_new.csv')
        df_sorted.to_csv(backup_csv, index=False)
        print(f"\n[WARNING] Could not write to {output_csv} (file locked). Saved to backup instead: {backup_csv}")
    
    # --- SAVE THE TOP 2 ML/DL MODELS ---
    print("\n[INFO] Saving the top 2 machine learning models based on Macro F1...")
    
    # Filter for candidate models to save (only ML/DL, skip Hybrid and Heuristics)
    candidates = []
    for idx, row in df_sorted.iterrows():
        model_name = row['model_name']
        feature_type = row['feature_type']
        config = row['config']
        macro_f1 = row['macro_f1']
        
        # Skip hybrid or baseline static, we only save the actual trainable model files
        if "Hybrid" in model_name or "Baseline" in model_name:
            continue
            
        # Try to find the trained model object
        model_obj = None
        is_keras = False
        
        if feature_type == 'advanced_12feature':
            # Reconstruct key for advanced features dict
            model_type_key = model_name.replace("ML Adv - ", "")
            key = f"{model_type_key} ({config})" if config != 'default' else model_type_key
            if key in ml_adv_models:
                model_obj = ml_adv_models[key]
        elif feature_type == 'raw_7frame_ear':
            # Reconstruct key for raw features dict
            model_type_key = model_name.replace("ML Raw - ", "")
            key = f"{model_type_key} ({config})" if config != 'default' else model_type_key
            if key in ml_raw_models:
                model_obj = ml_raw_models[key]
        elif feature_type == '3d_temporal_dynamic' and "LSTM" in model_name:
            model_obj = lstm_model
            is_keras = True
            
        if model_obj is not None:
            candidates.append({
                'model': model_obj,
                'name': model_name,
                'config': config,
                'macro_f1': macro_f1,
                'is_keras': is_keras
            })
            
    # Save the top 2 candidates
    save_dir = os.path.join('dataset_master', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    for i, candidate in enumerate(candidates[:2]):
        model_obj = candidate['model']
        model_name = candidate['name']
        config = candidate['config']
        macro_f1 = candidate['macro_f1']
        is_keras = candidate['is_keras']
        
        # File path naming: 1st best is 'best_traditional_model', 2nd best is 'second_best_traditional_model'
        rank_str = "best" if i == 0 else "second_best"
        ext = ".h5" if is_keras else ".pkl"
        filename = f"{rank_str}_traditional_model{ext}"
        filepath = os.path.join(save_dir, filename)
        
        if is_keras:
            model_obj.save(filepath)
        else:
            joblib.dump(model_obj, filepath)
            
        print(f"  Rank {i+1}: {model_name} ({config}) -> Saved to {filepath} (Macro F1: {macro_f1:.4f})")
        
if __name__ == '__main__':
    run_all_benchmarks()
