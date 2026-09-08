"""
attention_matrix_analysis.py — Fase 6, Tujuan 2: Analisis matriks attention.

Analisis:
1. Ekstraksi attention map (softmax(QK^T/√d)) sebelum/sesudah LoRA
2. Entropi attention per head per layer
3. Heatmap overlay ke citra daun (visualisasi fokus attention)
4. Perbandingan pola attention backbone-only vs backbone+LoRA

Usage:
    python -m src.analysis.attention_matrix_analysis --config configs/analysis.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path


def extract_attention_maps(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str = "cuda"
) -> List[torch.Tensor]:
    """Ekstraksi attention maps dari semua layer model.
    
    Args:
        model: ViT model (CassavaClassifier atau ViTModel).
        images: Input images of shape (batch, 3, H, W).
        device: Device.
        
    Returns:
        List of attention tensors per layer,
        each of shape (batch, num_heads, seq_len, seq_len).
    """
    model.eval()
    images = images.to(device)
    
    with torch.no_grad():
        if hasattr(model, "backbone"):
            outputs = model.backbone(
                pixel_values=images,
                output_attentions=True
            )
        else:
            outputs = model(
                pixel_values=images,
                output_attentions=True
            )
    
    # attentions: tuple of tensors, one per layer
    attention_maps = [attn.cpu() for attn in outputs.attentions]
    
    return attention_maps


def compute_attention_entropy(
    attention_maps: List[torch.Tensor]
) -> Dict[str, np.ndarray]:
    """Hitung entropi attention per head per layer.
    
    Entropi tinggi = attention tersebar merata (kurang fokus)
    Entropi rendah = attention fokus pada sedikit token
    
    Args:
        attention_maps: List of attention tensors per layer.
        
    Returns:
        Dict berisi entropy per layer dan per head.
    """
    layer_entropies = []
    head_entropies = []
    
    for layer_idx, attn in enumerate(attention_maps):
        # attn: (batch, heads, seq_len, seq_len)
        # Fokus pada baris CLS token (index 0)
        cls_attention = attn[:, :, 0, :]  # (batch, heads, seq_len)
        
        # Hitung entropi: H = -sum(p * log(p))
        # Tambah epsilon kecil untuk menghindari log(0)
        eps = 1e-10
        entropy = -(cls_attention * torch.log(cls_attention + eps)).sum(dim=-1)
        
        # Mean entropy per head (rata-rata over batch)
        mean_entropy_per_head = entropy.mean(dim=0).numpy()  # (heads,)
        head_entropies.append(mean_entropy_per_head)
        
        # Mean entropy per layer (rata-rata over batch dan heads)
        mean_entropy_layer = entropy.mean().item()
        layer_entropies.append(mean_entropy_layer)
    
    return {
        "layer_entropy": np.array(layer_entropies),  # (num_layers,)
        "head_entropy": np.array(head_entropies),     # (num_layers, num_heads)
    }


def compute_attention_distance(
    attn_before: List[torch.Tensor],
    attn_after: List[torch.Tensor]
) -> Dict[str, np.ndarray]:
    """Hitung jarak/perbedaan attention maps sebelum vs sesudah LoRA.
    
    Metrik:
    - Jensen-Shannon divergence per layer
    - Cosine similarity per layer
    - L2 distance per layer
    
    Args:
        attn_before: Attention maps dari model pretrained.
        attn_after: Attention maps dari model + LoRA.
        
    Returns:
        Dict berisi distance metrics per layer.
    """
    js_divs = []
    cos_sims = []
    l2_dists = []
    
    for before, after in zip(attn_before, attn_after):
        # Fokus pada CLS attention
        cls_before = before[:, :, 0, :].mean(dim=0)  # (heads, seq_len) — avg over batch
        cls_after = after[:, :, 0, :].mean(dim=0)
        
        # Jensen-Shannon divergence
        m = 0.5 * (cls_before + cls_after)
        eps = 1e-10
        kl_bm = (cls_before * torch.log((cls_before + eps) / (m + eps))).sum(dim=-1)
        kl_am = (cls_after * torch.log((cls_after + eps) / (m + eps))).sum(dim=-1)
        jsd = 0.5 * (kl_bm + kl_am)
        js_divs.append(jsd.mean().item())
        
        # Cosine similarity
        cos = F.cosine_similarity(
            cls_before.flatten().unsqueeze(0),
            cls_after.flatten().unsqueeze(0)
        )
        cos_sims.append(cos.item())
        
        # L2 distance
        l2 = torch.norm(cls_before - cls_after).item()
        l2_dists.append(l2)
    
    return {
        "js_divergence": np.array(js_divs),
        "cosine_similarity": np.array(cos_sims),
        "l2_distance": np.array(l2_dists),
    }


def generate_attention_heatmap_data(
    attention_maps: List[torch.Tensor],
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    image_size: int = 224,
    patch_size: int = 16
) -> np.ndarray:
    """Generate data heatmap attention untuk overlay ke citra.
    
    Args:
        attention_maps: List of attention tensors.
        layer_idx: Index layer (-1 untuk layer terakhir).
        head_idx: Index head (None = rata-rata semua head).
        image_size: Ukuran citra input.
        patch_size: Ukuran patch ViT.
        
    Returns:
        Heatmap array of shape (H_patches, W_patches).
    """
    attn = attention_maps[layer_idx]  # (batch, heads, seq_len, seq_len)
    
    if head_idx is not None:
        cls_attn = attn[:, head_idx, 0, 1:]  # Skip CLS token itself
    else:
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # Average over heads
    
    # Average over batch
    cls_attn = cls_attn.mean(dim=0).numpy()  # (num_patches,)
    
    # Reshape ke 2D grid
    num_patches_side = image_size // patch_size  # 14 for 224/16
    heatmap = cls_attn.reshape(num_patches_side, num_patches_side)
    
    # Normalize ke [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
    return heatmap


def main():
    parser = argparse.ArgumentParser(description="Attention matrix analysis (T2)")
    parser.add_argument("--config", type=str, default="configs/analysis.yaml")
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model_mae_vit.yaml")
    set_seed(config["global"]["seed"])
    
    device = config["global"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    attn_config = config["attention_analysis"]
    
    print("[attention_analysis] Loading models...")
    
    # Load pretrained backbone
    from src.models.mae_vit import load_pretrained_snapshot, load_mae_vit_backbone
    from src.models.classifier_head import CassavaClassifier, build_classifier_head
    from src.models.lora_layers import attach_lora_peft
    from src.models.lora_layers import rebuild_lora_backbone
    
    pretrained_path = resolve_path(config["checkpoints"]["pretrained_backbone"])
    backbone_pretrained = load_pretrained_snapshot(str(pretrained_path), device)
    
    # Load LoRA model
    lora_key = f"r{args.rank}"
    lora_ckpt_path = resolve_path(config["checkpoints"]["lora"][lora_key])
    
    backbone_lora = load_mae_vit_backbone(
      model_name=model_config["backbone"]["model_name"],
      pretrained=False, device=device
    )

    backbone_lora = rebuild_lora_backbone(backbone_lora, rank=args.rank)  # target_key default "qv"

    head_config = model_config["classifier"].copy()
    head_config["hidden_size"] = model_config["backbone"]["architecture"]["hidden_size"]
    head = build_classifier_head(head_config)

    model_lora = CassavaClassifier(backbone_lora, head).to(device)
    lora_ckpt = torch.load(str(lora_ckpt_path), map_location=device)
    model_lora.load_state_dict(lora_ckpt["model_state_dict"])
    
    # Load test data (sample)
    from src.data.dataset import CassavaDataset, get_val_test_transforms
    
    splits_dir = resolve_path(data_config["dataset"]["splits_dir"])
    transform = get_val_test_transforms(data_config)
    test_dataset = CassavaDataset(
        str(splits_dir / "test.csv"), transform=transform,
        class_names=data_config["dataset"]["classes"]
    )
    
    num_samples = attn_config.get("num_samples", 10)
    indices = list(range(min(num_samples * 5, len(test_dataset))))
    
    # Collect sample images
    sample_images = []
    for idx in indices[:num_samples]:
        img, _ = test_dataset[idx]
        sample_images.append(img)
    
    sample_batch = torch.stack(sample_images)
    
    # Extract attention maps
    print("[attention_analysis] Extracting attention maps (pretrained)...")
    attn_pretrained = extract_attention_maps(backbone_pretrained, sample_batch, device)
    
    print("[attention_analysis] Extracting attention maps (LoRA)...")
    attn_lora = extract_attention_maps(model_lora, sample_batch, device)
    
    # Compute entropy
    entropy_pretrained = compute_attention_entropy(attn_pretrained)
    entropy_lora = compute_attention_entropy(attn_lora)
    
    print("\n[attention_analysis] Entropy per layer (pretrained vs LoRA):")
    for i in range(len(entropy_pretrained["layer_entropy"])):
        print(f"  Layer {i}: {entropy_pretrained['layer_entropy'][i]:.4f} → "
              f"{entropy_lora['layer_entropy'][i]:.4f}")
    
    # Compute attention distance
    distance = compute_attention_distance(attn_pretrained, attn_lora)
    
    print("\n[attention_analysis] Attention distance (pretrained vs LoRA):")
    for i in range(len(distance["js_divergence"])):
        print(f"  Layer {i}: JSD={distance['js_divergence'][i]:.4f}, "
              f"CosSim={distance['cosine_similarity'][i]:.4f}")
    
    # Generate figures
    try:
        from src.viz.plotting import (
            plot_attention_entropy_comparison,
            plot_attention_heatmap_overlay
        )
        
        figures_config = attn_config["figures"]
        
        plot_attention_entropy_comparison(
            entropy_pretrained, entropy_lora,
            save_path=str(resolve_path(figures_config["attention_entropy"]))
        )
        
        # Heatmap overlay (first sample, last layer)
        heatmap_pre = generate_attention_heatmap_data(attn_pretrained)
        heatmap_lora = generate_attention_heatmap_data(attn_lora)
        
        plot_attention_heatmap_overlay(
            sample_images[0], heatmap_pre, heatmap_lora,
            save_path=str(resolve_path(figures_config["attention_heatmap"]))
        )
        
        print("[attention_analysis] Figures generated.")
    except ImportError as e:
        print(f"[attention_analysis] WARNING: Could not generate figures: {e}")
    
    print("\n[attention_analysis] Done.")


if __name__ == "__main__":
    main()
