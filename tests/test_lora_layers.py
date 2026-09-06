"""
test_lora_layers.py — Unit tests untuk LoRA implementation.

Tests:
- LinearWithLoRA output shape
- Parameter counting (trainable vs frozen)
- ΔW computation
- Freeze logic
"""

import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lora_layers import LinearWithLoRA


# ============================================================
# Test: LinearWithLoRA
# ============================================================

class TestLinearWithLoRA:
    """Tests untuk custom LoRA layer."""
    
    @pytest.fixture
    def lora_layer(self):
        """Buat LinearWithLoRA layer."""
        original = nn.Linear(768, 768)
        return LinearWithLoRA(original, rank=8, alpha=16.0, dropout=0.0)
    
    def test_output_shape(self, lora_layer):
        """Output shape harus sama dengan original linear."""
        x = torch.randn(4, 197, 768)
        output = lora_layer(x)
        assert output.shape == (4, 197, 768)
    
    def test_original_frozen(self, lora_layer):
        """Bobot original harus frozen."""
        assert not lora_layer.original_linear.weight.requires_grad
        if lora_layer.original_linear.bias is not None:
            assert not lora_layer.original_linear.bias.requires_grad
    
    def test_lora_trainable(self, lora_layer):
        """Bobot LoRA harus trainable."""
        assert lora_layer.lora_A.requires_grad
        assert lora_layer.lora_B.requires_grad
    
    def test_lora_a_shape(self, lora_layer):
        """Matrix A harus shape (rank, in_features)."""
        assert lora_layer.lora_A.shape == (8, 768)
    
    def test_lora_b_shape(self, lora_layer):
        """Matrix B harus shape (out_features, rank)."""
        assert lora_layer.lora_B.shape == (768, 8)
    
    def test_lora_b_initialized_zero(self):
        """Matrix B harus initialized dengan zeros."""
        original = nn.Linear(768, 768)
        layer = LinearWithLoRA(original, rank=8, alpha=16.0)
        assert torch.all(layer.lora_B == 0)
    
    def test_initial_output_matches_original(self):
        """Saat B=0, output LoRA layer harus sama dengan original."""
        torch.manual_seed(42)
        original = nn.Linear(768, 768)
        # Clone untuk perbandingan
        original_clone = nn.Linear(768, 768)
        original_clone.load_state_dict(original.state_dict())
        
        layer = LinearWithLoRA(original, rank=8, alpha=16.0, dropout=0.0)
        
        x = torch.randn(2, 768)
        
        with torch.no_grad():
            out_lora = layer(x)
            out_original = original_clone(x)
        
        torch.testing.assert_close(out_lora, out_original, atol=1e-6, rtol=1e-5)
    
    def test_delta_w_shape(self, lora_layer):
        """ΔW harus shape (out_features, in_features)."""
        delta_w = lora_layer.get_delta_w()
        assert delta_w.shape == (768, 768)
    
    def test_delta_w_computation(self):
        """ΔW = B·A * (alpha/rank) harus benar."""
        original = nn.Linear(64, 64)
        layer = LinearWithLoRA(original, rank=4, alpha=8.0, dropout=0.0)
        
        # Set known values
        with torch.no_grad():
            layer.lora_A.fill_(1.0)
            layer.lora_B.fill_(1.0)
        
        delta_w = layer.get_delta_w()
        # B·A: (64, 4) @ (4, 64) = (64, 64), all entries = 4
        # Scaled: 4 * (8/4) = 8
        expected_value = 4.0 * (8.0 / 4.0)
        assert torch.allclose(delta_w, torch.full_like(delta_w, expected_value))
    
    def test_get_lora_weights(self, lora_layer):
        """get_lora_weights harus mengembalikan dict yang lengkap."""
        weights = lora_layer.get_lora_weights()
        
        assert "A" in weights
        assert "B" in weights
        assert "delta_W" in weights
        assert "rank" in weights
        assert "alpha" in weights
        assert weights["rank"] == 8
        assert weights["alpha"] == 16.0
    
    def test_parameter_count(self, lora_layer):
        """Jumlah parameter trainable harus = rank * (in + out)."""
        trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        # A: 8 * 768 = 6144, B: 768 * 8 = 6144
        expected = 8 * 768 + 768 * 8  # = 12288
        assert trainable == expected


# ============================================================
# Test: Different Ranks
# ============================================================

class TestLoRARanks:
    """Test LoRA dengan berbagai rank."""
    
    @pytest.mark.parametrize("rank", [1, 4, 8, 16, 32, 64])
    def test_different_ranks(self, rank):
        """LoRA harus bekerja dengan berbagai rank."""
        original = nn.Linear(768, 768)
        layer = LinearWithLoRA(original, rank=rank, alpha=rank * 2)
        
        x = torch.randn(2, 768)
        output = layer(x)
        
        assert output.shape == (2, 768)
        assert layer.lora_A.shape[0] == rank
        assert layer.lora_B.shape[1] == rank
    
    @pytest.mark.parametrize("in_features,out_features", [
        (768, 768), (768, 3072), (3072, 768), (64, 32)
    ])
    def test_different_dimensions(self, in_features, out_features):
        """LoRA harus bekerja dengan dimensi berbeda."""
        original = nn.Linear(in_features, out_features)
        layer = LinearWithLoRA(original, rank=4, alpha=8.0)
        
        x = torch.randn(2, in_features)
        output = layer(x)
        
        assert output.shape == (2, out_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
