"""
config.py
---------
Configuration parameters, directory mappings, label encodings,
and hyperparameter settings for Cataract Grading and Dilation tasks.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathConfig:
    SEVERITY_CSV_PATH: str = "/kaggle/working/data/Dataset_Severity_V2.csv"
    SEVERITY_IMG_DIR: str = (
        "/kaggle/working/data/Dataset_Severity_V2/Dataset_Severity_V2"
    )

    DILATION_CSV_PATH: str = "/kaggle/working/data/Dataset_poordilation.csv"
    DILATION_IMG_DIR: str = (
        "/kaggle/working/data/Dataset_poordilation/Dataset_poordilation"
    )

    OUTPUT_ROOT_DIR: str = "/kaggle/working/outputs"



@dataclass
class TaskConfig:
    # -------------------------------------------------------------
    # Label Encodings
    # -------------------------------------------------------------
    SEVERITY_LABEL_MAP: Dict[str, int] = field(
        default_factory=lambda: {
            "low": 0,
            "dense": 1,
            "mature": 2,
            "brunescent": 3,
        }
    )

    DILATION_LABEL_MAP: Dict[str, int] = field(
        default_factory=lambda: {
            "normal": 0,
            "poor dilation": 1,
        }
    )

    # -------------------------------------------------------------
    # Class Balancing Quotas (per-epoch samples with replacement)
    # -------------------------------------------------------------
    SEVERITY_TARGET_PER_CLASS: Dict[int, int] = field(
        default_factory=lambda: {
            0: 1200,  # low
            1: 1200,  # dense
            2: 1200,  # mature
            3: 400,  # brunescent
        }
    )

    DILATION_TARGET_PER_CLASS: Dict[int, int] = field(
        default_factory=lambda: {
            0: 1000,  # normal
            1: 1000,  # poor dilation
        }
    )

    # -------------------------------------------------------------
    # Loss Weights
    # -------------------------------------------------------------
    SEVERITY_LOSS_WEIGHTS: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 2.0]
    )
    DILATION_LOSS_WEIGHTS: List[float] = field(
        default_factory=lambda: [1.0, 1.0]
    )


@dataclass
class ModelConfig:
    # Standard training resolution per architecture family
    FAMILY_RESOLUTIONS: Dict[str, int] = field(
        default_factory=lambda: {
            "resnet": 224,
            "densenet": 224,
            "efficientnet": 260,
        }
    )

    BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 30
    NUM_WORKERS: int = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)
    RANDOM_STATE: int = 42

    # Optuna HPO parameters
    OPTUNA_TRIALS: int = 12
    OPTUNA_EPOCHS: int = 8