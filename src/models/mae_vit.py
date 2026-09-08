"""
mae_vit.py — Load dan konfigurasi backbone MAE-ViT.

Mendukung:
- Load checkpoint pretrained dari HuggingFace (facebook/vit-mae-base)
- Extract encoder dari MAE model
- Freeze/unfreeze parameter backbone
- Simpan snapshot backbone pretrained (untuk analisis sebelum/sesudah LoRA)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any

from transformers import ViTMAEModel, ViTMAEConfig, ViTModel, ViTConfig


def load_mae_vit_backbone(
    model_name: str = "facebook/vit-mae-base",
    pretrained: bool = True,
    device: Optional[str] = None
) -> nn.Module:
    """Load MAE-ViT backbone dari HuggingFace.
    
    MAE pretrained model di-load, lalu encoder-nya diambil sebagai backbone
    untuk classification. Kita menggunakan ViTModel (bukan ViTMAEModel)
    karena untuk downstream task kita hanya butuh encoder.
    
    Args:
        model_name: HuggingFace model identifier.
        pretrained: Apakah load pretrained weights.
        device: Device target ('cuda', 'cpu').
        
    Returns:
        ViTModel backbone.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[mae_vit.py] Loading backbone: {model_name}")
    print(f"[mae_vit.py] Pretrained: {pretrained}")
    print(f"[mae_vit.py] Device: {device}")
    
    if pretrained:
        # Load MAE pretrained model
        mae_model = ViTMAEModel.from_pretrained(model_name, attn_implementation="eager")
        
        # Ambil config dan buat ViT model untuk downstream
        vit_config = ViTConfig(
            hidden_size=mae_model.config.hidden_size,
            num_hidden_layers=mae_model.config.num_hidden_layers,
            num_attention_heads=mae_model.config.num_attention_heads,
            intermediate_size=mae_model.config.intermediate_size,
            image_size=mae_model.config.image_size,
            patch_size=mae_model.config.patch_size,
            num_channels=mae_model.config.num_channels,
        )
        vit_config.attn_implementation = "eager" 
        vit_config._attn_implementation = "eager"
        vit_model = ViTModel(vit_config)
        
        # Transfer weights dari MAE encoder ke ViT model
        # MAE dan ViT berbagi arsitektur encoder yang sama
        mae_state_dict = mae_model.state_dict()
        vit_state_dict = vit_model.state_dict()
        
        # Filter dan map weights yang kompatibel
        compatible_weights = {}
        for key in vit_state_dict:
            if key in mae_state_dict and vit_state_dict[key].shape == mae_state_dict[key].shape:
                compatible_weights[key] = mae_state_dict[key]
        
        # Load compatible weights
        missing, unexpected = vit_model.load_state_dict(compatible_weights, strict=False)
        print(f"[mae_vit.py] Weights transferred: {len(compatible_weights)}/{len(vit_state_dict)}")
        if missing:
            print(f"[mae_vit.py] Missing keys (randomly initialized): {len(missing)}")
        
        del mae_model  # Free memory
    else:
        vit_config = ViTConfig()
        vit_config.attn_implementation = "eager"
        vit_config._attn_implementation = "eager"
        vit_model = ViTModel(vit_config)
        print("[mae_vit.py] Initialized with random weights")
    
    vit_model = vit_model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in vit_model.parameters())
    print(f"[mae_vit.py] Total parameters: {total_params:,}")
    print(f"[mae_vit.py] Hidden size: {vit_model.config.hidden_size}")
    print(f"[mae_vit.py] Num layers: {vit_model.config.num_hidden_layers}")
    print(f"[mae_vit.py] Num attention heads: {vit_model.config.num_attention_heads}")
    
    return vit_model


def freeze_backbone(model: nn.Module) -> None:
    """Bekukan semua parameter backbone (untuk linear probe / LoRA).
    
    Args:
        model: ViT model.
    """
    for param in model.parameters():
        param.requires_grad = False
    
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"[mae_vit.py] Frozen: {frozen}/{total} parameter tensors")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze semua parameter backbone (untuk full fine-tuning).
    
    Args:
        model: ViT model.
    """
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"[mae_vit.py] Unfrozen: {trainable}/{total} parameter tensors")


def save_pretrained_snapshot(model: nn.Module, save_path: str) -> None:
    """Simpan snapshot backbone pretrained SEBELUM fine-tuning/LoRA.
    
    Penting untuk analisis sebelum/sesudah (T2, T3) — memastikan
    perbandingan apple-to-apple.
    
    Args:
        model: ViT backbone model.
        save_path: Path untuk menyimpan checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict() if hasattr(model.config, "to_dict") else {},
    }, str(save_path))
    
    print(f"[mae_vit.py] Pretrained snapshot saved to {save_path}")


def load_pretrained_snapshot(save_path: str, device: Optional[str] = None) -> nn.Module:
    """Load snapshot backbone pretrained dari file.
    
    Args:
        save_path: Path ke checkpoint file.
        device: Device target.
        
    Returns:
        ViTModel dengan weights pretrained.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(str(save_path), map_location=device)
    
    config = ViTConfig(**checkpoint["config"]) if checkpoint["config"] else ViTConfig()
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"
    model = ViTModel(config)  # <-- dan ini
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    print(f"[mae_vit.py] Pretrained snapshot loaded from {save_path}")
    return model


def get_backbone_info(model: nn.Module) -> Dict[str, Any]:
    """Dapatkan informasi backbone untuk logging.
    
    Args:
        model: ViT model.
        
    Returns:
        Dict berisi info model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_pct": round(trainable_params / total_params * 100, 2) if total_params > 0 else 0,
        "hidden_size": model.config.hidden_size if hasattr(model, "config") else None,
        "num_layers": model.config.num_hidden_layers if hasattr(model, "config") else None,
    }
