"""
preprocessing.py — Preprocessing pipeline: stratified split, distribusi kelas, validasi data.

Usage:
    python -m src.data.preprocessing --config configs/data.yaml

Output:
    - data/splits/train.csv, val.csv, test.csv
    - reports/tables/class_distribution.csv
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Tambah project root ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def collect_image_paths(raw_dir: Path, classes: list, class_labels: dict) -> pd.DataFrame:
    """Kumpulkan semua path gambar dan label dari folder dataset.
    
    Args:
        raw_dir: Path ke folder dataset (berisi subfolder per kelas).
        classes: List nama kelas.
        class_labels: Dict mapping nama kelas ke integer label.
        
    Returns:
        DataFrame dengan kolom [image_path, class_name, label].
    """
    records = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    for class_name in classes:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: Folder kelas '{class_name}' tidak ditemukan di {raw_dir}")
            continue
        
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in supported_extensions:
                records.append({
                    "image_path": str(img_path),
                    "class_name": class_name,
                    "label": class_labels[class_name]
                })
    
    df = pd.DataFrame(records)
    print(f"[preprocessing.py] Total images found: {len(df)}")
    return df


def create_stratified_splits(
    df: pd.DataFrame, 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float,
    seed: int
) -> tuple:
    """Buat stratified split train/val/test.
    
    Args:
        df: DataFrame dengan kolom 'label'.
        train_ratio: Proporsi training set.
        val_ratio: Proporsi validation set.
        test_ratio: Proporsi test set.
        seed: Random seed.
        
    Returns:
        Tuple (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, val_test_df = train_test_split(
        df, 
        test_size=val_test_ratio, 
        random_state=seed, 
        stratify=df["label"]
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=val_test_df["label"]
    )
    
    return train_df, val_df, test_df


def compute_class_distribution(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    classes: list
) -> pd.DataFrame:
    """Hitung distribusi kelas per split.
    
    Returns:
        DataFrame dengan distribusi kelas untuk setiap split.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    rows = []
    
    for split_name, df in splits.items():
        counter = Counter(df["class_name"])
        for cls in classes:
            count = counter.get(cls, 0)
            total = len(df)
            rows.append({
                "split": split_name,
                "class_name": cls,
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0
            })
    
    return pd.DataFrame(rows)


def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> np.ndarray:
    """Hitung class weights untuk mengatasi imbalance.
    
    Menggunakan formula: weight_c = N_total / (N_classes * N_c)
    
    Args:
        train_df: Training DataFrame.
        num_classes: Jumlah kelas.
        
    Returns:
        Array of class weights.
    """
    counter = Counter(train_df["label"])
    total = len(train_df)
    weights = np.zeros(num_classes)
    
    for label, count in counter.items():
        weights[label] = total / (num_classes * count)
    
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Cassava dataset: stratified split & class distribution"
    )
    parser.add_argument(
        "--config", type=str, default="configs/data.yaml",
        help="Path to data config YAML"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    seed = config["split"]["seed"]
    set_seed(seed)
    
    # Resolve paths
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    splits_dir = resolve_path(config["dataset"]["splits_dir"])
    reports_path = resolve_path(config["reports"]["class_distribution"])
    
    classes = config["dataset"]["classes"]
    class_labels = config["dataset"]["class_labels"]
    
    print(f"[preprocessing.py] Raw data dir: {raw_dir}")
    print(f"[preprocessing.py] Splits output: {splits_dir}")
    
    # 1. Kumpulkan semua image paths
    df = collect_image_paths(raw_dir, classes, class_labels)
    
    if len(df) == 0:
        print("[preprocessing.py] ERROR: Tidak ada gambar ditemukan!")
        print(f"  Pastikan dataset ada di: {raw_dir}")
        sys.exit(1)
    
    # 2. Buat stratified split
    print(f"\n[preprocessing.py] Creating stratified split "
          f"({config['split']['train_ratio']}/{config['split']['val_ratio']}/{config['split']['test_ratio']})...")
    
    train_df, val_df, test_df = create_stratified_splits(
        df,
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        seed=seed
    )
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # 3. Simpan split CSV
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    print(f"\n[preprocessing.py] Splits saved to {splits_dir}")
    
    # 4. Hitung dan simpan distribusi kelas
    dist_df = compute_class_distribution(train_df, val_df, test_df, classes)
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(reports_path, index=False)
    print(f"[preprocessing.py] Class distribution saved to {reports_path}")
    
    # 5. Hitung class weights
    num_classes = len(classes)
    weights = compute_class_weights(train_df, num_classes)
    print(f"\n[preprocessing.py] Class weights (for weighted loss/sampler):")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {weights[i]:.4f}")
    
    # Simpan weights ke file
    weights_df = pd.DataFrame({
        "class_name": classes,
        "label": [class_labels[c] for c in classes],
        "weight": weights
    })
    weights_path = reports_path.parent / "class_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"[preprocessing.py] Class weights saved to {weights_path}")
    
    # Print ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN DISTRIBUSI KELAS")
    print("=" * 60)
    pivot = dist_df.pivot(index="class_name", columns="split", values="count")
    pivot = pivot[["train", "val", "test"]]  # reorder columns
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string())
    print("=" * 60)
    
    print("\n[preprocessing.py] Done. Fase 1 data pipeline selesai.")


if __name__ == "__main__":
    main()
