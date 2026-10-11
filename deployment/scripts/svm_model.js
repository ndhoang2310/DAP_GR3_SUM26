// File JavaScript thuần được tạo tự động bởi m2cgen từ best_traditional_model.pkl
// Model: Linear SVM (C=1.0) bao gồm cả StandardScaler


// Hàm chuẩn hóa dữ liệu (Mô phỏng StandardScaler)
function scaleFeatures(features) {
    const means = [0.6752469272296807, 0.6737017924602868, 0.6725220800182549, 0.6714380202314604, 0.6702953510151061, 0.6691685511922274, 0.6679594091891049, 0.507868499276569, 0.7969124411797837, 0.106623576604414, 1.1889615924343448, -0.6957993747328436];
    const scales = [0.2713902768227525, 0.2719001010346916, 0.2720850184390759, 0.2721467612058563, 0.27234243298509575, 0.27245381260934654, 0.2724463863953083, 0.2990028349562022, 0.21182428948940185, 0.10333687204930131, 0.8506178463328083, 0.9678806777945306];
    let scaled = [];
    for (let i = 0; i < features.length; i++) {
        let s = scales[i] === 0 ? 1 : scales[i];
        scaled.push((features[i] - means[i]) / s);
    }
    return scaled;
}

// Hàm dự đoán SVM tuyến tính (Hỗ trợ Multiclass OVO)
function scoreSVM(scaledFeatures) {
    const weightsMatrix = [[-1.4853157095623715, 0.08323188269022808, 0.7651512682633808, -0.036299116800734055, 0.2797655229840359, 0.7779974773880554, -1.729817412798468, 3.3474356117500292, -1.011851936515491, 1.576183147437888, -0.24204545474600003, 0.06144285727999943], [0.1201771588875431, 0.9421966466919685, 1.0594604838037753, 1.0070309945169242, 0.9265913121052165, 0.9189209472346249, 0.4052476790695039, -1.1129057242573737, -2.295774617556617, 1.291423392134483, 0.5059260896031788, 0.08368987599553179], [1.7378706496294711, 0.4685411967207074, 0.43751351278956463, 0.5125088086967651, 0.782124750418518, 0.4459600037034761, 1.7865891870063422, -5.581133086434733, -0.8328222861781143, -0.7303044313476903, -0.0726979525233844, -0.39067986970530555]];
    const biasArray = [0.8821322930832338, 1.5892444299656163, -0.18277126929983975];
    const classes = [0, 1, 2];
    const numClasses = classes.length;
    
    // Khởi tạo biến đếm phiếu (votes) cho mỗi class
    let votes = new Array(numClasses).fill(0);
    
    let k = 0;
    // Lặp qua tất cả các cặp (i, j)
    for (let i = 0; i < numClasses; i++) {
        for (let j = i + 1; j < numClasses; j++) {
            let score = biasArray[k];
            for (let f = 0; f < scaledFeatures.length; f++) {
                score += scaledFeatures[f] * weightsMatrix[k][f];
            }
            
            // Bỏ phiếu
            if (score > 0) {
                votes[i]++;
            } else {
                votes[j]++;
            }
            k++;
        }
    }
    
    // Tìm class có nhiều phiếu nhất
    let maxVotes = -1;
    let predictedClass = -1;
    for (let i = 0; i < numClasses; i++) {
        if (votes[i] > maxVotes) {
            maxVotes = votes[i];
            predictedClass = classes[i];
        }
    }
    
    return predictedClass;
}

// Hàm dự đoán chính (wrapper)
// Trả về trực tiếp class dự đoán (0: Mở mắt, 1: Nháy mắt, 2: Ngủ gật)
function predictPipeline(features) {
    const scaledFeatures = scaleFeatures(features);
    return scoreSVM(scaledFeatures);
}
