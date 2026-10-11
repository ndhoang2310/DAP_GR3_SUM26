"""
Central Configuration for Eye Blink Data Collection.
All constants, paths, and thresholds are defined here.
"""

import os

# ========================================================
# PATHS
# ========================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. Nguồn dữ liệu gốc (Master - chỉ đọc)
MASTER_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_master")
RAW_EYES_DIR = os.path.join(MASTER_DATASET_DIR, "raw_eyes")

# 2. Nơi lưu dữ liệu đã qua xử lý (Split - nơi script sẽ ghi vào)
PROCESSED_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_split")
TRAIN_DIR = os.path.join(PROCESSED_DATASET_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATASET_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATASET_DIR, "test")

# 3. Nơi lưu Model
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# CAMERA SETTINGS
# ============================================================
CAMERA_INDEX = 0
CAMERA_FPS = 15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ============================================================
# EYE PATCH SETTINGS
# ============================================================
EYE_PATCH_SIZE = (24, 24)  # Width x Height for cropped eye images

# ============================================================
# EAR (Eye Aspect Ratio) SETTINGS
# ============================================================
# EAR threshold for semi-auto labeling (used in label_tool.py)
EAR_THRESHOLD_LABELING = 0.09

# ============================================================
# DATASET SPLIT SETTINGS
# ============================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ========================================================
# TRAINING SETTINGS
# ========================================================
# Dành cho cả ML và DL
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
CLASS_WEIGHTS = {0: 1, 1: 6} # Trọng số cho lớp 0 (mở) và 1 (nhắm) dựa trên tỷ lệ dữ liệu của bạn

# Dành riêng cho Deep Learning
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
