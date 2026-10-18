"""
Configuration file for AD-MRI SSL Research Project

All settings in one place for reproducibility.
Change values here, not in the code.
"""

import os
from pathlib import Path

# ============================================================================
# REPRODUCIBILITY (Critical for research)
# ============================================================================

RANDOM_SEED = 42  # Set random seed for numpy, torch, etc.
DETERMINISTIC = True  # Force deterministic behavior (slightly slower but reproducible)

# ============================================================================
# PROJECT PATHS (Store data locally, never upload to cloud)
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Create directories if they don't exist
for dir_path in [PROCESSED_DATA_DIR, SPLITS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET SETTINGS
# ============================================================================

# Source dataset (for training)
SOURCE_DATASET = "adni"  # "adni" or "mock" (for testing)

# Target dataset (for cross-cohort evaluation)
TARGET_DATASET = "oasis"  # "oasis" or "mock"

# Class labels (Alzheimer's disease classification)
CLASS_NAMES = {0: "CN", 1: "MCI", 2: "AD"}  # CN=Cognitively Normal, MCI=Mild Cognitive Impairment, AD=Alzheimer's Disease
NUM_CLASSES = 3

# ============================================================================
# DATA SPLITTING (Subject-level, NOT slice-level!)
# ============================================================================

# CRITICAL: Split by SUBJECT, not by slice
# All slices from one subject must stay in the same split
TRAIN_RATIO = 0.70  # 70% of subjects for training
VAL_RATIO = 0.15    # 15% of subjects for validation
TEST_RATIO = 0.15   # 15% of subjects for testing

# ============================================================================
# MRI PREPROCESSING
# ============================================================================

# Image dimensions after preprocessing
MRI_SIZE = 128  # Resize slices to 128x128 pixels

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Architecture
MODEL_NAME = "resnet18"  # ResNet-18 (lightweight, proven)
PRETRAINED_IMAGENET = False  # Start from scratch or use ImageNet weights?

# Device
DEVICE = "cpu"  # "cpu" or "cuda" (we detected CUDA is not available, using CPU)

# ============================================================================
# TRAINING - SUPERVISED BASELINE
# ============================================================================

SUPERVISED_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 16,  # Small batch size for CPU
    "epochs": 50,
    "early_stopping_patience": 10,  # Stop if validation doesn't improve for 10 epochs
    "optimizer": "adam",
    "loss": "cross_entropy",
}

# ============================================================================
# TRAINING - SELF-SUPERVISED LEARNING (SimCLR)
# ============================================================================

SSL_CONFIG = {
    # Pretraining stage (unsupervised)
    "pretrain_epochs": 50,
    "pretrain_learning_rate": 0.001,
    "pretrain_batch_size": 16,
    
    # Fine-tuning stage (supervised)
    "finetune_epochs": 50,
    "finetune_learning_rate": 0.0001,
    "finetune_batch_size": 16,
    "early_stopping_patience": 10,
    
    # SimCLR-specific
    "temperature": 0.5,  # Temperature for contrastive loss
    "projection_dim": 128,  # Dimension of projection head
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

# Important: Augmentations ONLY on training data, not validation/test
AUGMENTATION_CONFIG = {
    "rotation_range": 15,  # Degrees
    "elastic_deformation": True,
    "intensity_jitter": 0.1,  # ±10% intensity scaling
    "gaussian_noise_std": 0.05,
}

# ============================================================================
# EVALUATION
# ============================================================================

# Multiple runs for statistical significance
NUM_RUNS = 5  # Run experiment 5 times with different seeds

# Metrics to compute
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "balanced_accuracy",
    "confusion_matrix",
]

# ============================================================================
# LOGGING & SAVING
# ============================================================================

SAVE_CHECKPOINTS = True  # Save model weights during training
SAVE_PREDICTIONS = True  # Save subject-level predictions
VERBOSE = True  # Print progress messages

# ============================================================================
# PRINT CONFIGURATION (for verification)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROJECT CONFIGURATION")
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Deterministic: {DETERMINISTIC}")
    print(f"Device: {DEVICE}")
    print(f"Source Dataset: {SOURCE_DATASET}")
    print(f"Target Dataset: {TARGET_DATASET}")
    print(f"Train/Val/Test Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"MRI Size: {MRI_SIZE}x{MRI_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print("=" * 70)