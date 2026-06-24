import tensorflow as tf
from tensorflow.keras import layers, models

def get_cnn_model(input_shape=(24, 24, 1)):
    """
    Kiến trúc CNN tối ưu cho bài toán nhận diện đóng/mở mắt.
    """
    data_aug = tf.keras.Sequential([
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(
            0.05,
            0.05
        )
    ])
    model = models.Sequential([
        layers.Input(shape=input_shape),

        data_aug,
        # Tầng tích chập 1: Thu nhận các đặc trưng cơ bản
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Tầng tích chập 2: Thu nhận các đặc trưng phức tạp hơn
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten và các tầng Dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3), # Giảm thiểu Overfitting
        layers.Dense(1, activation='sigmoid') # Đầu ra 0 (open) hoặc 1 (closed)
    ])
    return model