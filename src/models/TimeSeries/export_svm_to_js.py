import os
import joblib
import m2cgen as m2c

def main():
    print("Bắt đầu convert model...")
    # Đường dẫn file model tốt nhất
    model_path = os.path.join('dataset_master', 'models', 'best_traditional_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy model tại {model_path}")
        return
        
    # Load model
    model = joblib.load(model_path)
    print(f"Đã tải model từ: {model_path}")
    print(f"Kiểu model: {type(model)}")
    
    # Xử lý Pipeline: Trích xuất thủ công do m2cgen gặp lỗi đệ quy với SVC
    if hasattr(model, 'steps'):
        scaler = model.steps[0][1]
        svm_model = model.steps[1][1]
        
        # Trích xuất tham số của StandardScaler
        means = scaler.mean_.tolist()
        scales = scaler.scale_.tolist()
        
        # Trích xuất tham số của Linear SVM OVO (One-vs-One cho 3 class: 0vs1, 0vs2, 1vs2)
        if hasattr(svm_model, 'coef_'):
            coefs = svm_model.coef_.tolist()
            intercept = svm_model.intercept_.tolist()
            classes = svm_model.classes_.tolist()
        else:
            coefs = (svm_model.dual_coef_ @ svm_model.support_vectors_).tolist()
            intercept = svm_model.intercept_.tolist()
            classes = svm_model.classes_.tolist()
        
        # Nối code Scaler và SVM lại
        js_code = f"""
// Hàm chuẩn hóa dữ liệu (Mô phỏng StandardScaler)
function scaleFeatures(features) {{
    const means = {means};
    const scales = {scales};
    let scaled = [];
    for (let i = 0; i < features.length; i++) {{
        let s = scales[i] === 0 ? 1 : scales[i];
        scaled.push((features[i] - means[i]) / s);
    }}
    return scaled;
}}

// Hàm dự đoán SVM tuyến tính (Hỗ trợ Multiclass OVO)
function scoreSVM(scaledFeatures) {{
    const weightsMatrix = {coefs};
    const biasArray = {intercept};
    const classes = {classes};
    const numClasses = classes.length;
    
    // Khởi tạo biến đếm phiếu (votes) cho mỗi class
    let votes = new Array(numClasses).fill(0);
    
    let k = 0;
    // Lặp qua tất cả các cặp (i, j)
    for (let i = 0; i < numClasses; i++) {{
        for (let j = i + 1; j < numClasses; j++) {{
            let score = biasArray[k];
            for (let f = 0; f < scaledFeatures.length; f++) {{
                score += scaledFeatures[f] * weightsMatrix[k][f];
            }}
            
            // Bỏ phiếu
            if (score > 0) {{
                votes[i]++;
            }} else {{
                votes[j]++;
            }}
            k++;
        }}
    }}
    
    // Tìm class có nhiều phiếu nhất
    let maxVotes = -1;
    let predictedClass = -1;
    for (let i = 0; i < numClasses; i++) {{
        if (votes[i] > maxVotes) {{
            maxVotes = votes[i];
            predictedClass = classes[i];
        }}
    }}
    
    return predictedClass;
}}

// Hàm dự đoán chính (wrapper)
// Trả về trực tiếp class dự đoán (0: Mở mắt, 1: Nháy mắt, 2: Ngủ gật)
function predictPipeline(features) {{
    const scaledFeatures = scaleFeatures(features);
    return scoreSVM(scaledFeatures);
}}
"""
    else:
        print("Mô hình không phải là Pipeline, không hỗ trợ.")
        return
        
    # Tạo thư mục deployment/web_demo nếu chưa có
    deploy_dir = os.path.join('src', 'deployment', 'web_demo')
    os.makedirs(deploy_dir, exist_ok=True)
    
    out_js_path = os.path.join(deploy_dir, 'svm_model.js')
    
    with open(out_js_path, 'w', encoding='utf-8') as f:
        f.write("// File JavaScript thuần được tạo tự động bởi m2cgen từ best_traditional_model.pkl\n")
        f.write("// Model: Linear SVM (C=1.0) bao gồm cả StandardScaler\n\n")
        f.write(js_code)
        
    size_kb = os.path.getsize(out_js_path) / 1024
    print(f"[THÀNH CÔNG] Đã lưu mã nguồn JavaScript tại: {out_js_path} (Dung lượng: {size_kb:.2f} KB)")

if __name__ == '__main__':
    main()
