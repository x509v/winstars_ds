"""
Split a single-folder ImageFolder layout into train/val for image_train.py and EDA.

Your data is currently:
    data/animals/animals/<class>/*.jpg

This script creates:
    data/animals/train/<class>/*.jpg   (default 80%)
    data/animals/val/<class>/*.jpg     (default 20%)

Usage (from task2_animal_ner_vision):
    python split_animals_dataset.py --source data/animals/animals --target data/animals --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split animal class folders into train/val.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/animals",
        help="Root folder containing one subfolder per class (e.g. data/animals/animals).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/animals",
        help="Root where train/ and val/ will be created.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of images per class for validation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    random.seed(args.seed)
    train_root = target / "train"
    val_root = target / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        if not images:
            continue
        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        val_images = images[:n_val]
        train_images = images[n_val:]

        (train_root / name).mkdir(exist_ok=True)
        (val_root / name).mkdir(exist_ok=True)
        for f in train_images:
            shutil.copy2(f, train_root / name / f.name)
        for f in val_images:
            shutil.copy2(f, val_root / name / f.name)
        print(f"{name}: {len(train_images)} train, {len(val_images)} val")

    print(f"Done. Train: {train_root}, Val: {val_root}")


if __name__ == "__main__":
    main()
