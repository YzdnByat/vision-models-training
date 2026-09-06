"""
dataset.py
----------
Dataset abstractions, patient-wise (video-wise) splitting with zero-leakage
validation, balanced class sampling, and PyTorch DataLoader factories.
"""

import os
import multiprocessing
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import GroupShuffleSplit

from config import PathConfig, TaskConfig, ModelConfig
from transforms import get_transforms


class BalancedClassSampler(Sampler):
    """
    Ensures a fixed quota of samples per class per epoch.
    Sampling is performed with replacement.
    """
    def __init__(self, labels: np.ndarray, target_per_class: Dict[int, int]):
        self.labels = np.asarray(labels)
        self.target_per_class = target_per_class
        self.classes = np.unique(self.labels)

        self.class_indices = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

        # Verification check
        for c in self.classes:
            if c not in self.target_per_class:
                raise ValueError(f"Target quota missing for class {c}")

    def __iter__(self):
        indices = []
        for c in self.classes:
            target_n = self.target_per_class[c]
            chosen = np.random.choice(
                self.class_indices[c], target_n, replace=True
            )
            indices.extend(chosen.tolist())

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return sum(self.target_per_class.values())


class CataractDataset(Dataset):
    """
    Standard PyTorch dataset reading images from patient video subdirectories.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        label_map: Dict[str, int],
        transform=None
    ):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = os.path.join(
            self.image_dir, str(row["videoname"]), str(row["filename"])
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file at: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class PatientDataSplitter:
    """
    Performs patient-isolated (video-isolated) 70/15/15 train/val/test splits.
    """
    @staticmethod
    def load_and_prep_df(task: str, csv_path: str, img_dir: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if task == "severity":
            df["label"] = None
            df.loc[df["low nuclear density"] == 1.0, "label"] = "low"
            df.loc[df["dense"] == 1.0, "label"] = "dense"
            df.loc[df["mature"] == 1.0, "label"] = "mature"
            df.loc[df["brunescent"] == 1.0, "label"] = "brunescent"
            df = df[df["label"].notna()].reset_index(drop=True)
        elif task == "poor_dilation":
            # Direct categorical label column
            if "label" not in df.columns:
                raise KeyError("Expected 'label' column in dilation CSV.")
        else:
            raise ValueError(f"Unsupported task: {task}")

        # Map filename to parent video folder if videoname is absent or flat
        if "videoname" not in df.columns:
            filename_to_video = {}
            for vname in os.listdir(img_dir):
                vpath = os.path.join(img_dir, vname)
                if os.path.isdir(vpath):
                    for fname in os.listdir(vpath):
                        filename_to_video[fname] = vname
            df["videoname"] = df["filename"].map(filename_to_video)

        missing_vids = df["videoname"].isna().sum()
        if missing_vids > 0:
            raise AssertionError(f"{missing_vids} images missing videoname mapping.")

        return df[["filename", "videoname", "label"]].copy()

    @classmethod
    def split(
        cls, df: pd.DataFrame, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Stage 1: 70% Train, 30% Temp (Val + Test)
        gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
        train_idx, temp_idx = next(gss_train.split(df, groups=df["videoname"]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)

        # Stage 2: 50% Val, 50% Test of Temp (15% / 15% overall)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
        val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["videoname"]))

        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        # Enforce strict patient-isolation assertions
        train_vids = set(train_df["videoname"])
        val_vids = set(val_df["videoname"])
        test_vids = set(test_df["videoname"])

        assert len(train_vids & val_vids) == 0, "Patient leakage between Train and Val sets!"
        assert len(train_vids & test_vids) == 0, "Patient leakage between Train and Test sets!"
        assert len(val_vids & test_vids) == 0, "Patient leakage between Val and Test sets!"

        return train_df, val_df, test_df


def build_dataloaders(
    task: str,
    img_size: int,
    batch_size: int = ModelConfig.BATCH_SIZE,
    num_workers: int = ModelConfig.NUM_WORKERS,
    csv_path: Optional[str] = None,
    img_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, pd.DataFrame]]:
    """
    Prepares patient-split data and returns PyTorch DataLoaders.
    """
    if task == "severity":
        csv_file = csv_path or PathConfig.SEVERITY_CSV_PATH
        img_folder = img_dir or PathConfig.SEVERITY_IMG_DIR
        label_map = TaskConfig.SEVERITY_LABEL_MAP
        target_per_class = TaskConfig.SEVERITY_TARGET_PER_CLASS
    elif task == "poor_dilation":
        csv_file = csv_path or PathConfig.DILATION_CSV_PATH
        img_folder = img_dir or PathConfig.DILATION_IMG_DIR
        label_map = TaskConfig.DILATION_LABEL_MAP
        target_per_class = TaskConfig.DILATION_TARGET_PER_CLASS
    else:
        raise ValueError(f"Unknown task: {task}")

    # Process and split metadata
    df = PatientDataSplitter.load_and_prep_df(task, csv_file, img_folder)
    train_df, val_df, test_df = PatientDataSplitter.split(df, random_state=ModelConfig.RANDOM_STATE)

    # Transforms
    train_transform, eval_transform = get_transforms(task=task, img_size=img_size)

    # Datasets
    train_dataset = CataractDataset(train_df, img_folder, label_map, transform=train_transform)
    val_dataset = CataractDataset(val_df, img_folder, label_map, transform=eval_transform)
    test_dataset = CataractDataset(test_df, img_folder, label_map, transform=eval_transform)

    # Balanced Sampler for training split
    y_train = train_df["label"].map(label_map).to_numpy()
    train_sampler = BalancedClassSampler(labels=y_train, target_per_class=target_per_class)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,  # Sampler manages randomization
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    return train_loader, val_loader, test_loader, split_dfs