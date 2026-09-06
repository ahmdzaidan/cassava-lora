"""
lora_layers.py — Implementasi LoRA (Low-Rank Adaptation).

Dua opsi implementasi:
1. Via HuggingFace PEFT library (direkomendasikan)
2. Custom LoRA layer (untuk pemahaman / sanity check)

Fitur:
- Attach LoRA ke proyeksi attention (Q, K, V, O) pada ViT encoder
- Konfigurasi rank, alpha, dropout via YAML
- Extract dan simpan bobot LoRA (A, B, ΔW) secara terpisah per layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from pathlib import Path
import copy


# ============================================================
# Opsi 1: Custom LoRA Layer (manual implementation)
# ============================================================

class LinearWithLoRA(nn.Module):
    """Linear layer dengan LoRA adapter.
    
    Original linear: y = Wx + b
    Dengan LoRA:     y = Wx + b + (BAx) * (alpha/r)
    
    Args:
        original_linear: Linear layer asli yang akan di-adapt.
        rank: Rank LoRA (r).
        alpha: Scaling factor LoRA.
        dropout: Dropout rate untuk LoRA.
    """
    
    def __init__(
        self, 
        original_linear: nn.Linear, 
        rank: int = 8, 
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        
        # LoRA matrices
        # A: down-projection (in_features → rank), initialized with Kaiming
        # B: up-projection (rank → out_features), initialized with zeros
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        
        # Optional dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA.
        
        Args:
            x: Input tensor of shape (..., in_features).
            
        Returns:
            Output tensor of shape (..., out_features).
        """
        # Original forward
        original_output = self.original_linear(x)
        
        # LoRA forward: (B @ A @ x^T)^T * scaling
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B) * self.scaling
        
        return original_output + lora_output
    
    def get_delta_w(self) -> torch.Tensor:
        """Hitung ΔW = B·A (scaled).
        
        Returns:
            Tensor ΔW of shape (out_features, in_features).
        """
        return (self.lora_B @ self.lora_A) * self.scaling
    
    def get_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Dapatkan bobot LoRA untuk analisis.
        
        Returns:
            Dict berisi A, B, dan delta_W.
        """
        return {
            "A": self.lora_A.data.clone(),
            "B": self.lora_B.data.clone(),
            "delta_W": self.get_delta_w().detach(),
            "rank": self.rank,
            "alpha": self.alpha,
        }


# ============================================================
# Opsi 2: Via HuggingFace PEFT (direkomendasikan)
# ============================================================

def attach_lora_peft(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none"
) -> nn.Module:
    """Attach LoRA ke model menggunakan HuggingFace PEFT.
    
    Args:
        model: ViT model (HuggingFace ViTModel).
        rank: LoRA rank.
        alpha: LoRA alpha (scaling).
        target_modules: List nama modul target (e.g., ["query", "value"]).
        lora_dropout: Dropout rate.
        bias: Bias training mode ("none", "all", "lora_only").
        
    Returns:
        Model dengan LoRA adapter (PEFT model).
    """
    from peft import LoraConfig, get_peft_model, TaskType
    
    if target_modules is None:
        target_modules = ["query", "value"]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        # Tidak pakai TaskType karena model bukan AutoModel for classification
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model


# ============================================================
# Fungsi Attach LoRA Custom (manual, tanpa PEFT)
# ============================================================

def attach_lora_custom(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05
) -> nn.Module:
    """Attach LoRA custom ke modul attention ViT.
    
    Mengganti nn.Linear di attention layers dengan LinearWithLoRA.
    
    Args:
        model: ViT model.
        rank: LoRA rank.
        alpha: LoRA alpha.
        target_modules: Nama-nama sub-module target di attention.
            Default: ["query", "value"] (proyeksi Q dan V).
        lora_dropout: Dropout rate.
        
    Returns:
        Model yang sama dengan LoRA layers ter-attach.
    """
    if target_modules is None:
        target_modules = ["query", "value"]
    
    lora_count = 0
    
    # Iterasi semua named modules
    for name, module in model.named_modules():
        # Cari attention layers
        for target in target_modules:
            if hasattr(module, target):
                original_linear = getattr(module, target)
                if isinstance(original_linear, nn.Linear):
                    lora_layer = LinearWithLoRA(
                        original_linear=original_linear,
                        rank=rank,
                        alpha=alpha,
                        dropout=lora_dropout
                    )
                    setattr(module, target, lora_layer)
                    lora_count += 1
    
    print(f"[lora_layers.py] Attached {lora_count} custom LoRA layers")
    print(f"[lora_layers.py] Rank: {rank}, Alpha: {alpha}, Dropout: {lora_dropout}")
    print(f"[lora_layers.py] Target modules: {target_modules}")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora_layers.py] Total params: {total_params:,}")
    print(f"[lora_layers.py] Trainable params: {trainable_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")
    
    return model


# ============================================================
# Utility: Extract & Save LoRA Weights
# ============================================================

def extract_lora_weights_peft(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract bobot LoRA dari PEFT model.
    
    Args:
        model: PEFT model dengan LoRA adapters.
        
    Returns:
        Dict {layer_name: {"A": tensor, "B": tensor, "delta_W": tensor}}.
    """
    lora_weights = {}
    
    for name, module in model.named_modules():
        # PEFT LoRA layers punya lora_A dan lora_B
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # PEFT stores lora_A/B as ModuleDicts with adapter name keys
            for adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.data.clone()
                B = module.lora_B[adapter_name].weight.data.clone()
                delta_W = B @ A
                
                key = f"{name}.{adapter_name}" if adapter_name != "default" else name
                lora_weights[key] = {
                    "A": A,
                    "B": B,
                    "delta_W": delta_W,
                }
    
    print(f"[lora_layers.py] Extracted LoRA weights from {len(lora_weights)} layers")
    return lora_weights


def extract_lora_weights_custom(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract bobot LoRA dari custom LinearWithLoRA layers.
    
    Args:
        model: Model dengan LinearWithLoRA layers.
        
    Returns:
        Dict {layer_name: {"A": tensor, "B": tensor, "delta_W": tensor}}.
    """
    lora_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            lora_weights[name] = module.get_lora_weights()
    
    print(f"[lora_layers.py] Extracted LoRA weights from {len(lora_weights)} layers")
    return lora_weights


def save_lora_weights(
    lora_weights: Dict[str, Dict[str, torch.Tensor]], 
    save_path: str
) -> None:
    """Simpan bobot LoRA ke file.
    
    Args:
        lora_weights: Dict LoRA weights dari extract_lora_weights_*.
        save_path: Path untuk menyimpan (*.pt).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors ke CPU untuk portabilitas
    cpu_weights = {}
    for layer_name, weights in lora_weights.items():
        cpu_weights[layer_name] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in weights.items()
        }
    
    torch.save(cpu_weights, str(save_path))
    print(f"[lora_layers.py] LoRA weights saved to {save_path}")


def load_lora_weights(load_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load bobot LoRA dari file.
    
    Args:
        load_path: Path ke file LoRA weights (*.pt).
        
    Returns:
        Dict LoRA weights.
    """
    weights = torch.load(str(load_path), map_location="cpu")
    print(f"[lora_layers.py] LoRA weights loaded from {load_path}")
    print(f"[lora_layers.py] Layers: {list(weights.keys())}")
    return weights
