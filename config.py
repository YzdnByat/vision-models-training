"""
config.py
---------
Configuration parameters, directory mappings, label encodings,
and hyperparameter settings for Cataract Grading and Dilation tasks.
"""

import os
from typing import Dict, List


class PathConfig:

    DATA_ROOT: str = "/kaggle/working/data"

    SEVERITY_CSV_PATH: str = os.path.join(
        DATA_ROOT,
        "Dataset_Severity",
        "SEV_metadata.csv"
    )

    SEVERITY_IMG_DIR: str = os.path.join(
        DATA_ROOT,
        "Dataset_Severity",
        "images"
    )

    DILATION_CSV_PATH: str = os.path.join(
        DATA_ROOT,
        "Dataset_Poordilation",
        "PD_metadata.csv"
    )

    DILATION_IMG_DIR: str = os.path.join(
        DATA_ROOT,
        "Dataset_Poordilation",
        "images"
    )

    OUTPUT_ROOT_DIR: str = "/kaggle/working/outputs"

class TaskConfig:

    # -------------------------------------------------------------
    # Label Encodings
    # -------------------------------------------------------------
    SEVERITY_LABEL_MAP: Dict[str, int] = {
        "low": 0,
        "dense": 1,
        "mature": 2,
        "brunescent": 3,
    }

    DILATION_LABEL_MAP: Dict[str, int] = {
        "normal": 0,
        "poor dilation": 1,
    }

    # -------------------------------------------------------------
    # Class Balancing Quotas
    # -------------------------------------------------------------
    SEVERITY_TARGET_PER_CLASS = {
        0: 550,  # low
        1: 700,  # dense
        2: 450,  # mature
        3: 150,  # brunescent
    }

    DILATION_TARGET_PER_CLASS: Dict[int, int] = {
        0: 1000,  # normal
        1: 1000,  # poor dilation
    }

    # -------------------------------------------------------------
    # Loss Weights
    # -------------------------------------------------------------
    SEVERITY_LOSS_WEIGHTS = [
        1.0,
        1.0,
        1.0,
        2.0,
    ]

    DILATION_LOSS_WEIGHTS: List[float] = [
        1.0,
        1.0,
    ]


class ModelConfig:

    # -------------------------------------------------------------
    # Input resolutions
    # -------------------------------------------------------------
    FAMILY_RESOLUTIONS: Dict[str, int] = {
        "resnet": 224,
        "densenet": 224,
        "efficientnet": 260,
    }

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------
    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 30

    NUM_WORKERS: int = max(
        1,
        os.cpu_count() // 2 if os.cpu_count() else 2
    )

    RANDOM_STATE: int = 42

    # -------------------------------------------------------------
    # Optuna
    # -------------------------------------------------------------
    OPTUNA_TRIALS: int = 12
    OPTUNA_EPOCHS: int = 8