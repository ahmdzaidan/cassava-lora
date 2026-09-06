"""
test_dataset.py — Unit tests untuk data pipeline.

Tests:
- Split integrity (jumlah, proporsi, no overlap)
- Distribusi kelas per split
- Dataset class functionality
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    create_stratified_splits,
    compute_class_distribution,
    compute_class_weights
)


# ============================================================
# Test: Stratified Split
# ============================================================

class TestStratifiedSplit:
    """Tests untuk create_stratified_splits."""
    
    @pytest.fixture
    def sample_df(self):
        """Buat DataFrame sampel untuk testing."""
        np.random.seed(42)
        n = 1000
        # Distribusi imbalanced (mirip dataset asli)
        labels = np.concatenate([
            np.full(100, 0),   # cbb
            np.full(300, 1),   # cbsd
            np.full(150, 2),   # cgm
            np.full(400, 3),   # cmd
            np.full(50, 4),    # healthy
        ])
        df = pd.DataFrame({
            "image_path": [f"img_{i}.jpg" for i in range(n)],
            "class_name": ["cbb"]*100 + ["cbsd"]*300 + ["cgm"]*150 + ["cmd"]*400 + ["healthy"]*50,
            "label": labels,
        })
        return df
    
    def test_split_sizes(self, sample_df):
        """Split sizes harus sesuai proporsi."""
        train, val, test = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=42
        )
        
        total = len(sample_df)
        assert abs(len(train) - total * 0.7) <= 5  # toleransi ±5
        assert abs(len(val) - total * 0.15) <= 5
        assert abs(len(test) - total * 0.15) <= 5
        assert len(train) + len(val) + len(test) == total
    
    def test_no_overlap(self, sample_df):
        """Tidak boleh ada overlap antar splits."""
        train, val, test = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=42
        )
        
        train_paths = set(train["image_path"])
        val_paths = set(val["image_path"])
        test_paths = set(test["image_path"])
        
        assert len(train_paths & val_paths) == 0
        assert len(train_paths & test_paths) == 0
        assert len(val_paths & test_paths) == 0
    
    def test_stratified_distribution(self, sample_df):
        """Distribusi kelas harus mirip di setiap split."""
        train, val, test = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=42
        )
        
        original_dist = sample_df["label"].value_counts(normalize=True)
        train_dist = train["label"].value_counts(normalize=True)
        
        # Toleransi 5% per kelas
        for label in original_dist.index:
            assert abs(train_dist.get(label, 0) - original_dist[label]) < 0.05, \
                f"Class {label} distribution differs > 5%"
    
    def test_reproducibility(self, sample_df):
        """Seed yang sama harus menghasilkan split yang sama."""
        train1, val1, test1 = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=42
        )
        train2, val2, test2 = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=42
        )
        
        assert train1["image_path"].tolist() == train2["image_path"].tolist()
        assert val1["image_path"].tolist() == val2["image_path"].tolist()
    
    def test_different_seeds_different_splits(self, sample_df):
        """Seed berbeda harus menghasilkan split berbeda."""
        train1, _, _ = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=42
        )
        train2, _, _ = create_stratified_splits(
            sample_df, 0.7, 0.15, 0.15, seed=123
        )
        
        assert train1["image_path"].tolist() != train2["image_path"].tolist()
    
    def test_invalid_ratios(self, sample_df):
        """Ratio yang tidak sum ke 1.0 harus error."""
        with pytest.raises(AssertionError):
            create_stratified_splits(sample_df, 0.7, 0.2, 0.2, seed=42)


# ============================================================
# Test: Class Distribution
# ============================================================

class TestClassDistribution:
    """Tests untuk compute_class_distribution."""
    
    def test_distribution_output(self):
        """Output harus memiliki kolom yang benar."""
        classes = ["cbb", "cbsd", "cgm"]
        train = pd.DataFrame({"class_name": ["cbb", "cbsd", "cgm", "cbb"]})
        val = pd.DataFrame({"class_name": ["cbsd", "cgm"]})
        test = pd.DataFrame({"class_name": ["cbb"]})
        
        result = compute_class_distribution(train, val, test, classes)
        
        assert "split" in result.columns
        assert "class_name" in result.columns
        assert "count" in result.columns
        assert "percentage" in result.columns
        assert len(result) == 3 * 3  # 3 splits x 3 classes


# ============================================================
# Test: Class Weights
# ============================================================

class TestClassWeights:
    """Tests untuk compute_class_weights."""
    
    def test_weight_computation(self):
        """Kelas dengan sedikit sampel harus mendapat weight lebih tinggi."""
        df = pd.DataFrame({"label": [0]*100 + [1]*10 + [2]*50})
        weights = compute_class_weights(df, num_classes=3)
        
        assert weights[1] > weights[0]  # kelas minoritas weight lebih tinggi
        assert weights[1] > weights[2]
    
    def test_balanced_weights(self):
        """Kelas yang seimbang harus mendapat weight yang sama."""
        df = pd.DataFrame({"label": [0]*100 + [1]*100 + [2]*100})
        weights = compute_class_weights(df, num_classes=3)
        
        np.testing.assert_allclose(weights[0], weights[1], atol=1e-6)
        np.testing.assert_allclose(weights[1], weights[2], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
