"""
Dataset splitting, directory management and reproducibility utilities.
"""
import random
import shutil
import json
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import (
    CLASES,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_STATE
)


def seed_everything(seed: int = RANDOM_STATE):
    """Ensures deterministic behavior across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def preparar_dataset_splits(
    origen_dir: Path,
    destino_dir: Path,
    test_split: float = TEST_SPLIT,
    val_split: float = VALIDATION_SPLIT,
    seed: int = RANDOM_STATE
) -> Tuple[Path, Path, Path, Dict]:
    """
    Performs reproducible train/val/test split copying images by class.
    """
    origen_dir = Path(origen_dir)
    destino_dir = Path(destino_dir)

    train_dir = destino_dir / "train"
    val_dir = destino_dir / "val"
    test_dir = destino_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {}
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.tif", "*.tiff", "*.bmp")

    for clase in CLASES:
        clase_dir = origen_dir / clase
        if not clase_dir.exists():
            print(f"[WARN] Folder for class '{clase}' not found in {origen_dir}")
            continue

        imagenes = []
        for ext in extensions:
            imagenes.extend(list(clase_dir.glob(ext)))

        if not imagenes:
            print(f"[WARN] No images found for class '{clase}' in {clase_dir}")
            continue

        train_val, test = train_test_split(
            imagenes,
            test_size=test_split,
            random_state=seed,
            shuffle=True
        )

        relative_val_size = val_split / (1.0 - test_split)
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=seed
        )

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            target_folder = destino_dir / split_name / clase
            target_folder.mkdir(parents=True, exist_ok=True)
            for img_path in split_files:
                shutil.copy2(img_path, target_folder / img_path.name)

        stats[clase] = {
            "total": len(imagenes),
            "train": len(train),
            "val": len(val),
            "test": len(test)
        }
        print(f"Class '{clase}': Total={len(imagenes)} (Train={len(train)}, Val={len(val)}, Test={len(test)})")

    return train_dir, val_dir, test_dir, stats
