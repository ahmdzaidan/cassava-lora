"""
dataset.py — PyTorch Dataset dan DataLoader untuk Cassava Leaf Disease.

Fitur:
- Load gambar dari CSV split files
- Augmentasi terpisah untuk train vs val/test
- WeightedRandomSampler untuk mengatasi class imbalance
- Mendukung konfigurasi dari YAML
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


class CassavaDataset(Dataset):
    """PyTorch Dataset untuk Cassava Leaf Disease Classification.
    
    Args:
        csv_path: Path ke file CSV split (train.csv, val.csv, test.csv).
        transform: Torchvision transforms untuk augmentasi/preprocessing.
        class_names: List nama kelas (untuk referensi).
    """
    
    def __init__(
        self, 
        csv_path: str, 
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[list] = None
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_names = class_names or sorted(self.df["class_name"].unique().tolist())
        
        # Validasi
        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert "label" in self.df.columns, "CSV must have 'label' column"
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self) -> np.ndarray:
        """Dapatkan semua labels (untuk WeightedRandomSampler)."""
        return self.df["label"].values
    
    def get_class_counts(self) -> Dict[str, int]:
        """Hitung jumlah sampel per kelas."""
        counter = Counter(self.df["class_name"])
        return {cls: counter.get(cls, 0) for cls in self.class_names}


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Buat transform untuk training set (dengan augmentasi ringan).
    
    Args:
        config: Data config dictionary.
        
    Returns:
        Compose transform.
    """
    aug_config = config["augmentation"]["train"]
    prep_config = config["preprocessing"]
    
    transform_list = [
        transforms.RandomResizedCrop(
            size=aug_config["random_resized_crop"]["size"],
            scale=tuple(aug_config["random_resized_crop"]["scale"])
        ),
        transforms.RandomHorizontalFlip(p=aug_config["random_horizontal_flip"]),
        transforms.ColorJitter(
            brightness=aug_config["color_jitter"]["brightness"],
            contrast=aug_config["color_jitter"]["contrast"],
            saturation=aug_config["color_jitter"]["saturation"],
            hue=aug_config["color_jitter"]["hue"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=prep_config["mean"],
            std=prep_config["std"]
        )
    ]
    
    return transforms.Compose(transform_list)


def get_val_test_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Buat transform untuk validation/test set (tanpa augmentasi).
    
    Args:
        config: Data config dictionary.
        
    Returns:
        Compose transform.
    """
    aug_config = config["augmentation"]["val_test"]
    prep_config = config["preprocessing"]
    
    transform_list = [
        transforms.Resize(aug_config["resize"]),
        transforms.CenterCrop(aug_config["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=prep_config["mean"],
            std=prep_config["std"]
        )
    ]
    
    return transforms.Compose(transform_list)


def create_weighted_sampler(dataset: CassavaDataset) -> WeightedRandomSampler:
    """Buat WeightedRandomSampler untuk mengatasi class imbalance.
    
    Memberikan bobot lebih tinggi ke sampel dari kelas minoritas sehingga
    setiap kelas mendapat kesempatan sampling yang setara.
    
    Args:
        dataset: CassavaDataset instance.
        
    Returns:
        WeightedRandomSampler.
    """
    labels = dataset.get_labels()
    class_counts = Counter(labels)
    num_samples = len(labels)
    
    # Weight per kelas = total / (jumlah_kelas * jumlah_per_kelas)
    num_classes = len(class_counts)
    class_weights = {
        cls: num_samples / (num_classes * count) 
        for cls, count in class_counts.items()
    }
    
    # Weight per sample
    sample_weights = np.array([class_weights[label] for label in labels])
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler


def get_class_weights_tensor(dataset: CassavaDataset, num_classes: int) -> torch.Tensor:
    """Hitung class weights sebagai tensor (untuk weighted CrossEntropyLoss).
    
    Args:
        dataset: CassavaDataset instance.
        num_classes: Jumlah kelas.
        
    Returns:
        Tensor of class weights.
    """
    labels = dataset.get_labels()
    counter = Counter(labels)
    total = len(labels)
    
    weights = torch.zeros(num_classes)
    for label, count in counter.items():
        weights[label] = total / (num_classes * count)
    
    return weights


def create_dataloaders(
    config: Dict[str, Any],
    splits_dir: str,
    use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Buat DataLoaders untuk train, val, dan test.
    
    Args:
        config: Data config dictionary.
        splits_dir: Path ke folder splits (berisi train.csv, val.csv, test.csv).
        use_weighted_sampler: Gunakan WeightedRandomSampler untuk train.
        
    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    splits_dir = Path(splits_dir)
    dl_config = config["dataloader"]
    class_names = config["dataset"]["classes"]
    
    # Transforms
    train_transform = get_train_transforms(config)
    val_test_transform = get_val_test_transforms(config)
    
    # Datasets
    train_dataset = CassavaDataset(
        csv_path=str(splits_dir / "train.csv"),
        transform=train_transform,
        class_names=class_names
    )
    val_dataset = CassavaDataset(
        csv_path=str(splits_dir / "val.csv"),
        transform=val_test_transform,
        class_names=class_names
    )
    test_dataset = CassavaDataset(
        csv_path=str(splits_dir / "test.csv"),
        transform=val_test_transform,
        class_names=class_names
    )
    
    # Sampler untuk train (mengatasi imbalance)
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
        train_shuffle = False  # Sampler dan shuffle tidak bisa dipakai bersamaan
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_config["batch_size"],
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=dl_config["num_workers"],
        pin_memory=dl_config["pin_memory"],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_config["batch_size"],
        shuffle=False,
        num_workers=dl_config["num_workers"],
        pin_memory=dl_config["pin_memory"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=dl_config["batch_size"],
        shuffle=False,
        num_workers=dl_config["num_workers"],
        pin_memory=dl_config["pin_memory"]
    )
    
    print(f"[dataset.py] DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
