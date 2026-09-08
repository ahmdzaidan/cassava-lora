"""
cls_embedding_shift.py — Fase 7, Tujuan 3: Pergeseran vektor [CLS].

Analisis:
1. Ekstraksi [CLS] token di setiap layer untuk model pretrained vs LoRA
2. Hitung cosine similarity shift per sampel per layer
3. Analisis pola shift (delayed specialization pattern)
4. Korelasi shift vs akurasi per kelas

Usage:
    python -m src.analysis.cls_embedding_shift --config configs/analysis.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path


def extract_cls_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    max_samples: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Ekstrak [CLS] embedding di setiap layer.
    
    Args:
        model: CassavaClassifier atau ViTModel.
        dataloader: DataLoader.
        device: Device.
        max_samples: Maks sampel (0 = semua).
        
    Returns:
        Tuple:
        - List of numpy arrays per layer, each shape (n_samples, hidden_size)
        - Labels array shape (n_samples,)
    """
    model.eval()
    all_cls_per_layer = None
    all_labels = []
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            if max_samples > 0 and total >= max_samples:
                break
            
            images = images.to(device)
            
            if hasattr(model, "backbone"):
                outputs = model.backbone(
                    pixel_values=images,
                    output_hidden_states=True
                )
            else:
                outputs = model(
                    pixel_values=images,
                    output_hidden_states=True
                )
            
            hidden_states = outputs.hidden_states
            
            if all_cls_per_layer is None:
                all_cls_per_layer = [[] for _ in range(len(hidden_states))]
            
            for i, hs in enumerate(hidden_states):
                cls_token = hs[:, 0, :].cpu().numpy()
                all_cls_per_layer[i].append(cls_token)
            
            all_labels.extend(labels.numpy())
            total += images.shape[0]
    
    # Concatenate
    cls_embeddings = [
        np.concatenate(layer_cls, axis=0) for layer_cls in all_cls_per_layer
    ]
    labels = np.array(all_labels)
    
    if max_samples > 0:
        cls_embeddings = [emb[:max_samples] for emb in cls_embeddings]
        labels = labels[:max_samples]
    
    print(f"[cls_shift] Extracted CLS from {len(cls_embeddings)} layers, "
          f"{cls_embeddings[0].shape[0]} samples")
    
    return cls_embeddings, labels


def compute_cosine_shift(
    cls_pretrained: List[np.ndarray],
    cls_lora: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """Hitung cosine similarity shift per sampel per layer.
    
    Shift = 1 - cos(CLS_pretrained, CLS_LoRA)
    
    Args:
        cls_pretrained: CLS embeddings dari model pretrained per layer.
        cls_lora: CLS embeddings dari model LoRA per layer.
        
    Returns:
        Dict berisi shift metrics per layer.
    """
    num_layers = len(cls_pretrained)
    
    shifts_per_layer = []
    mean_shifts = []
    std_shifts = []
    
    for i in range(num_layers):
        pre = torch.tensor(cls_pretrained[i], dtype=torch.float32)
        lora = torch.tensor(cls_lora[i], dtype=torch.float32)
        
        # Cosine similarity per sample
        cos_sim = F.cosine_similarity(pre, lora, dim=1)
        shift = 1.0 - cos_sim  # shift: 0 = identik, 2 = berlawanan
        
        shifts_per_layer.append(shift.numpy())
        mean_shifts.append(shift.mean().item())
        std_shifts.append(shift.std().item())
    
    return {
        "shifts_per_layer": shifts_per_layer,  # List of arrays (n_samples,)
        "mean_shift_per_layer": np.array(mean_shifts),
        "std_shift_per_layer": np.array(std_shifts),
    }


def compute_shift_per_class(
    shifts_per_layer: List[np.ndarray],
    labels: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """Hitung rata-rata shift per kelas per layer.
    
    Args:
        shifts_per_layer: Shift values per layer.
        labels: Labels array.
        class_names: Nama kelas.
        
    Returns:
        DataFrame dengan kolom [layer, class_name, mean_shift, std_shift, count].
    """
    rows = []
    
    for layer_idx, shifts in enumerate(shifts_per_layer):
        for class_idx, class_name in enumerate(class_names):
            mask = labels == class_idx
            if mask.sum() > 0:
                class_shifts = shifts[mask]
                rows.append({
                    "layer": layer_idx,
                    "class_name": class_name,
                    "mean_shift": np.mean(class_shifts),
                    "std_shift": np.std(class_shifts),
                    "count": int(mask.sum()),
                })
    
    return pd.DataFrame(rows)


def correlate_shift_with_accuracy(
    mean_shift_per_class: Dict[str, float],
    accuracy_improvement_per_class: Dict[str, float],
    method: str = "spearman"
) -> Dict[str, float]:
    """Uji korelasi antara besar shift CLS dan peningkatan akurasi per kelas.
    
    Args:
        mean_shift_per_class: {class_name: mean_shift}.
        accuracy_improvement_per_class: {class_name: accuracy_improvement}.
        method: 'pearson' atau 'spearman'.
        
    Returns:
        Dict berisi correlation coefficient, p-value, dan interpretasi.
    """
    classes = sorted(set(mean_shift_per_class.keys()) & set(accuracy_improvement_per_class.keys()))
    
    if len(classes) < 3:
        return {
            "method": method,
            "correlation": None,
            "p_value": None,
            "note": "Terlalu sedikit kelas untuk uji korelasi yang valid"
        }
    
    shifts = [mean_shift_per_class[c] for c in classes]
    improvements = [accuracy_improvement_per_class[c] for c in classes]
    
    if method == "pearson":
        corr, p_value = stats.pearsonr(shifts, improvements)
    elif method == "spearman":
        corr, p_value = stats.spearmanr(shifts, improvements)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Interpretasi
    if p_value < 0.05:
        if corr > 0.7:
            interpretation = "Korelasi positif kuat dan signifikan"
        elif corr > 0.3:
            interpretation = "Korelasi positif moderat dan signifikan"
        elif corr > -0.3:
            interpretation = "Korelasi lemah tetapi signifikan"
        elif corr > -0.7:
            interpretation = "Korelasi negatif moderat dan signifikan"
        else:
            interpretation = "Korelasi negatif kuat dan signifikan"
    else:
        interpretation = "Korelasi tidak signifikan secara statistik"
    
    return {
        "method": method,
        "correlation": round(corr, 4),
        "p_value": round(p_value, 4),
        "n_classes": len(classes),
        "interpretation": interpretation,
        "classes": classes,
        "shifts": shifts,
        "improvements": improvements,
    }


def main():
    parser = argparse.ArgumentParser(description="CLS embedding shift analysis (T3)")
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
    
    cls_config = config["cls_shift_analysis"]
    class_names = data_config["dataset"]["classes"]
    
    # Load models
    from src.models.mae_vit import load_pretrained_snapshot, load_mae_vit_backbone
    from src.models.classifier_head import CassavaClassifier, build_classifier_head
    from src.data.dataset import CassavaDataset, get_val_test_transforms
    from torch.utils.data import DataLoader
    from src.models.lora_layers import rebuild_lora_backbone

    print("[cls_shift] Loading pretrained backbone...")
    pretrained_path = resolve_path(config["checkpoints"]["pretrained_backbone"])
    backbone_pretrained = load_pretrained_snapshot(str(pretrained_path), device)
    
    print("[cls_shift] Loading LoRA model...")
    lora_key = f"r{args.rank}"
    lora_ckpt_path = resolve_path(config["checkpoints"]["lora"][lora_key])
    
    backbone_lora = load_mae_vit_backbone(
        model_name=model_config["backbone"]["model_name"],
        pretrained=False, device=device
    )
    backbone_lora = rebuild_lora_backbone(backbone_lora, rank=args.rank)

    head_config = model_config["classifier"].copy()
    head_config["hidden_size"] = model_config["backbone"]["architecture"]["hidden_size"]
    head = build_classifier_head(head_config)
    model_lora = CassavaClassifier(backbone_lora, head).to(device)
    lora_ckpt = torch.load(str(lora_ckpt_path), map_location=device)
    model_lora.load_state_dict(lora_ckpt["model_state_dict"])
    
    # Load test data
    splits_dir = resolve_path(data_config["dataset"]["splits_dir"])
    transform = get_val_test_transforms(data_config)
    test_dataset = CassavaDataset(
        str(splits_dir / "test.csv"), transform=transform,
        class_names=class_names
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    max_samples = cls_config.get("num_samples", 0)
    
    # Extract CLS embeddings
    print("\n[cls_shift] Extracting CLS embeddings (pretrained)...")
    cls_pre, labels = extract_cls_embeddings(backbone_pretrained, test_loader, device, max_samples)
    
    print("[cls_shift] Extracting CLS embeddings (LoRA)...")
    cls_lora, _ = extract_cls_embeddings(model_lora, test_loader, device, max_samples)
    
    # Compute shifts
    print("\n[cls_shift] Computing cosine shifts...")
    shift_results = compute_cosine_shift(cls_pre, cls_lora)
    
    print("\nMean shift per layer:")
    for i, (mean, std) in enumerate(zip(
        shift_results["mean_shift_per_layer"],
        shift_results["std_shift_per_layer"]
    )):
        print(f"  Layer {i}: {mean:.6f} ± {std:.6f}")
    
    # Shift per class
    shift_per_class_df = compute_shift_per_class(
        shift_results["shifts_per_layer"], labels, class_names
    )
    
    # Save results
    figures_config = cls_config["figures"]
    shift_path = resolve_path(figures_config["shift_per_layer"])
    shift_path = Path(str(shift_path)).parent
    shift_path.mkdir(parents=True, exist_ok=True)
    
    # Save per-class shift data
    tables_dir = resolve_path(config["global"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    shift_per_class_df.to_csv(tables_dir / "cls_shift_per_class.csv", index=False)
    
    # Generate figures
    try:
        from src.viz.plotting import plot_cls_shift_per_layer
        
        plot_cls_shift_per_layer(
            shift_results,
            save_path=str(resolve_path(figures_config["shift_per_layer"]))
        )
        print("[cls_shift] Figures generated.")
    except ImportError as e:
        print(f"[cls_shift] WARNING: Could not generate figures: {e}")
    
    print("\n[cls_shift] Done.")


if __name__ == "__main__":
    main()
