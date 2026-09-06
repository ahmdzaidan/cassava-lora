"""
test_analysis_functions.py — Unit tests untuk fungsi analisis.

Tests:
- SVD computation & properties
- Cosine shift computation
- Linear CKA correctness
- Clustering metrics
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Test: SVD Analysis
# ============================================================

class TestSVDAnalysis:
    """Tests untuk SVD weight analysis functions."""
    
    def test_svd_analysis_output(self):
        """Output SVD harus lengkap dan valid."""
        from src.analysis.svd_weight_analysis import svd_analysis_single_layer
        import torch
        
        delta_w = torch.randn(64, 128)
        result = svd_analysis_single_layer(delta_w)
        
        assert "U" in result
        assert "S" in result
        assert "Vh" in result
        assert "S_normalized" in result
        assert "cumulative_energy" in result
        assert "effective_rank" in result
        assert "frobenius_norm" in result
    
    def test_svd_singular_values_descending(self):
        """Singular values harus menurun."""
        from src.analysis.svd_weight_analysis import svd_analysis_single_layer
        import torch
        
        delta_w = torch.randn(32, 32)
        result = svd_analysis_single_layer(delta_w)
        
        S = result["S"]
        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1] - 1e-6, "Singular values should be descending"
    
    def test_svd_normalized_starts_at_one(self):
        """Normalized singular values harus dimulai dari ~1."""
        from src.analysis.svd_weight_analysis import svd_analysis_single_layer
        import torch
        
        delta_w = torch.randn(32, 32) * 5.0
        result = svd_analysis_single_layer(delta_w)
        
        assert abs(result["S_normalized"][0] - 1.0) < 0.01
    
    def test_svd_cumulative_energy_monotone(self):
        """Cumulative energy harus monoton naik."""
        from src.analysis.svd_weight_analysis import svd_analysis_single_layer
        import torch
        
        delta_w = torch.randn(32, 32)
        result = svd_analysis_single_layer(delta_w)
        
        energy = result["cumulative_energy"]
        for i in range(len(energy) - 1):
            assert energy[i] <= energy[i + 1] + 1e-6
    
    def test_svd_cumulative_energy_ends_at_one(self):
        """Cumulative energy harus berakhir di ~1.0."""
        from src.analysis.svd_weight_analysis import svd_analysis_single_layer
        import torch
        
        delta_w = torch.randn(32, 32)
        result = svd_analysis_single_layer(delta_w)
        
        assert abs(result["cumulative_energy"][-1] - 1.0) < 0.01
    
    def test_svd_low_rank_matrix(self):
        """Matrix rank-rendah harus punya effective rank kecil."""
        from src.analysis.svd_weight_analysis import svd_analysis_single_layer
        import torch
        
        # Buat matrix rank-2
        A = torch.randn(64, 2)
        B = torch.randn(2, 128)
        low_rank = A @ B
        
        result = svd_analysis_single_layer(low_rank)
        
        assert result["effective_rank"] <= 3  # rank 2, threshold 99% energy


# ============================================================
# Test: Cosine Shift
# ============================================================

class TestCosineShift:
    """Tests untuk cosine shift computation."""
    
    def test_identical_embeddings_zero_shift(self):
        """Embedding identik harus menghasilkan shift = 0."""
        from src.analysis.cls_embedding_shift import compute_cosine_shift
        
        emb = [np.random.randn(100, 768).astype(np.float32)]
        shift = compute_cosine_shift(emb, emb)
        
        np.testing.assert_allclose(
            shift["mean_shift_per_layer"], [0.0], atol=1e-5
        )
    
    def test_orthogonal_embeddings_max_shift(self):
        """Embedding ortogonal harus menghasilkan shift = 1."""
        from src.analysis.cls_embedding_shift import compute_cosine_shift
        
        # Buat dua set embedding ortogonal
        n = 50
        d = 100
        emb1 = np.zeros((n, d), dtype=np.float32)
        emb2 = np.zeros((n, d), dtype=np.float32)
        emb1[:, 0] = 1.0
        emb2[:, 1] = 1.0
        
        shift = compute_cosine_shift([emb1], [emb2])
        
        np.testing.assert_allclose(
            shift["mean_shift_per_layer"], [1.0], atol=1e-5
        )
    
    def test_shift_range(self):
        """Shift harus berada di range [0, 2]."""
        from src.analysis.cls_embedding_shift import compute_cosine_shift
        
        emb1 = [np.random.randn(100, 768).astype(np.float32)]
        emb2 = [np.random.randn(100, 768).astype(np.float32)]
        
        shift = compute_cosine_shift(emb1, emb2)
        
        assert all(0 <= s <= 2 for s in shift["mean_shift_per_layer"])


# ============================================================
# Test: Linear CKA
# ============================================================

class TestLinearCKA:
    """Tests untuk Linear CKA implementation."""
    
    def test_cka_identical_representations(self):
        """CKA dengan representasi identik harus = 1."""
        from src.analysis.cka import linear_cka
        
        X = np.random.randn(50, 768)
        result = linear_cka(X, X)
        
        assert abs(result - 1.0) < 1e-5
    
    def test_cka_symmetric(self):
        """CKA harus simetris: CKA(X,Y) = CKA(Y,X)."""
        from src.analysis.cka import linear_cka
        
        X = np.random.randn(50, 768)
        Y = np.random.randn(50, 256)
        
        cka_xy = linear_cka(X, Y)
        cka_yx = linear_cka(Y, X)
        
        assert abs(cka_xy - cka_yx) < 1e-5
    
    def test_cka_range(self):
        """CKA harus berada di range [0, 1]."""
        from src.analysis.cka import linear_cka
        
        X = np.random.randn(50, 100)
        Y = np.random.randn(50, 200)
        
        result = linear_cka(X, Y)
        
        assert 0 <= result <= 1 + 1e-5
    
    def test_cka_matrix_shape(self):
        """CKA matrix harus punya shape yang benar."""
        from src.analysis.cka import compute_cka_matrix
        
        reps1 = [np.random.randn(30, 64) for _ in range(5)]
        reps2 = [np.random.randn(30, 64) for _ in range(5)]
        
        matrix, names = compute_cka_matrix(reps1, reps2)
        
        assert matrix.shape == (5, 5)


# ============================================================
# Test: Clustering Metrics
# ============================================================

class TestClusteringMetrics:
    """Tests untuk clustering metrics."""
    
    def test_well_separated_clusters(self):
        """Kluster terpisah jelas harus punya Silhouette tinggi, DBI rendah."""
        from src.analysis.clustering_metrics import compute_clustering_quality
        
        np.random.seed(42)
        # 3 kluster yang jelas terpisah
        cluster1 = np.random.randn(50, 10) + np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(50, 10) + np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(50, 10) + np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0])
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        labels = np.array([0]*50 + [1]*50 + [2]*50)
        
        result = compute_clustering_quality(embeddings, labels)
        
        assert result["silhouette_score"] > 0.5  # harus positif tinggi
        assert result["davies_bouldin_index"] < 1.0  # harus rendah
    
    def test_overlapping_clusters(self):
        """Kluster tumpang tindih harus punya Silhouette rendah."""
        from src.analysis.clustering_metrics import compute_clustering_quality
        
        np.random.seed(42)
        embeddings = np.random.randn(150, 10)  # random, no structure
        labels = np.array([0]*50 + [1]*50 + [2]*50)
        
        result = compute_clustering_quality(embeddings, labels)
        
        assert result["silhouette_score"] < 0.5  # seharusnya rendah


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
