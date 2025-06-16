"""
Configuration settings for the v2-train.py script.
"""
from typing import Optional

# ===== MASTER SWITCH =====
# Determines if the temporal model or a standard non-temporal model is trained.
MODEL_IS_TEMPORAL: bool = True 

# ===== TRAINING CONFIGURATION =====
NUM_EPOCHS: int = 50
BATCH_SIZE: int = 16
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
GRAD_CLIP_MAX_NORM: float = 1.0
# EARLY_STOPPING_PATIENCE: int = 5 # Optional: Patience for early stopping

# ===== DATASET CONFIGURATION =====
IMG_HEIGHT: int = 512
IMG_WIDTH: int = 640

# ===== TEMPORAL MODEL SPECIFIC SETTINGS =====
# These settings are primarily used if MODEL_IS_TEMPORAL is True.
BASE_TEMPORAL_SEQ_LEN: int = 5
MOTION_CALC_IN_CHANNELS: int = 3

# ===== DATASET SUBSET PERCENTAGES =====
TRAIN_SUBSET_PERCENTAGE: Optional[float] = 1.0
VAL_SUBSET_PERCENTAGE: Optional[float] = 1.0

# ===== DATALOADER CONFIGURATION =====
NUM_WORKERS: int = 4
SHUFFLE_TRAIN: bool = True
DROP_LAST_TRAIN: bool = True
DROP_LAST_VAL: bool = False

# ===== PATHS CONFIGURATION =====
IMG_FOLDER: str = "c:/data/processed_anti_uav_v2/images"
TRAIN_SEQ_FILE: str = "c:/data/processed_anti_uav_v2/sequences/train.txt"
VAL_SEQ_FILE: str = "c:/data/processed_anti_uav_v2/sequences/val.txt"

# Configuration file basenames (relative to `configs/rtdetr/` directory)
TEMPORAL_CONFIG_BASENAME: str = "rtdetr_dla34_6x_uav_temporal_fixed.yml"
NON_TEMPORAL_CONFIG_BASENAME: str = "rtdetr_dla34_6x_coco.yml"

# ===== OUTPUT DIRECTORY =====
# This is a format string, {model_type_str} will be replaced in the main script.
OUTPUT_DIR_FORMAT_STRING: str = "./output/uav_{model_type_str}_training_v2_train"

# ===== RESUME TRAINING CONFIGURATION =====
# Path to a .pth checkpoint file to resume training from. Set to None to disable.
# Example: "./output/uav_temporal_training_v2_train/best_temporal_model.pth"
RESUME_CHECKPOINT_PATH: Optional[str] = "output/uav_temporal_training_v2_train/epoch_40_temporal_model.pth"

# ===== LOGGING CONFIGURATION =====
LOG_BATCH_UPDATE_INTERVAL: int = 50  # Increased interval to reduce terminal flooding
