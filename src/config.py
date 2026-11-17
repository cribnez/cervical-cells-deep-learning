"""
Configuration parameters for Cervical Cell Deep Learning System.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "models"
DEFAULT_WEIGHTS_PATH = DEFAULT_WEIGHTS_DIR / "modelo_pesos.pkl"

# Bethesda Classification Classes (Strict ordering)
CLASES = ["HSIL", "LSIL", "NILM"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training Hyperparameters
EPOCHS_FASE1 = 20
EPOCHS_FASE2 = 20
LR_INICIAL = 1e-3
LR_FINE = 3e-5

# Dataset Partition Splits
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42

# Transfer Learning & Fine-Tuning
UNFREEZE_LAST_N = 10
LABEL_SMOOTHING = 0.03

# Triage Asymmetric Class Weighting (Prioritize HSIL Sensitivity)
# 0: HSIL, 1: LSIL, 2: NILM
CLASS_WEIGHT = {0: 2.0, 1: 1.2, 2: 0.8}

# Data Augmentation & Color Normalization Probabilities
P_REINHARD_TRAIN = 0.70
P_GRAYSCALE = 0.15
HUE_DEG = 8
SAT_RANGE = (0.75, 1.35)
VAL_RANGE = (0.75, 1.25)
GAMMA_RANGE = (0.8, 1.3)

# Reinhard Color Normalization (LAB Space Targets)
TARGET_MEANS = (145, 128, 128)
TARGET_STDS = (20, 12, 12)

# Tissue-Aware Secure Cropping
CHROMA_THR_LOOSE = 6
L_DARK_THR_LOOSE = 245
MIN_TISSUE_AREA = 2500
MIN_TISSUE_FRAC = 0.08
MIN_CROP_SIDE = 160
PAD_EXTRA_FRAC = 0.15
CROP_ZOOM_MIN = 0.80
CROP_ZOOM_MAX = 1.00

# Clinical GUI Configuration
COLORES = {
    "HSIL": "#e74c3c",
    "LSIL": "#f39c12",
    "NILM": "#27ae60"
}
DEFAULT_NILM_THRESHOLD = 0.90
GRADCAM_LAYER_NAME = "top_conv"
