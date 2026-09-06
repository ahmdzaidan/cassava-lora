"""
clustering_metrics.py — Fase 7, Tujuan 3: Silhouette Score & Davies-Bouldin Index.

Mengukur kualitas kluster [CLS] embedding per kelas penyakit,
sebelum vs sesudah LoRA.

Usage:
    python -m src.analysis.clustering_metrics --config configs/analysis.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path


def compute_clustering_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """Hitung Silhouette Score dan Davies-Bouldin Index.
    
    Args:
        embeddings: Array of shape (n_samples, n_features).
        labels: Cluster/class labels of shape (n_samples,).
        prefix: Prefix untuk nama metrik.
        
    Returns:
        Dict berisi metrik kluster.
    """
    # Pastikan ada minimal 2 kelas
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            f"{prefix}silhouette_score": None,
            f"{prefix}davies_bouldin_index": None,
            f"{prefix}note": "Less than 2 classes present"
        }
    
    # Silhouette Score: [-1, 1], semakin tinggi semakin baik
    sil_score = silhouette_score(embeddings, labels)
    
    # Davies-Bouldin Index: >= 0, semakin rendah semakin baik
    dbi = davies_bouldin_score(embeddings, labels)
    
    return {
        f"{prefix}silhouette_score": round(sil_score, 4),
        f"{prefix}davies_bouldin_index": round(dbi, 4),
    }


def compare_clustering_before_after(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> pd.DataFrame:
    """Bandingkan kualitas kluster sebelum vs sesudah LoRA.
    
    Args:
        embeddings_before: CLS embeddings pretrained.
        embeddings_after: CLS embeddings setelah LoRA.
        labels: Class labels.
        class_names: Nama kelas.
        
    Returns:
        DataFrame perbandingan.
    """
    metrics_before = compute_clustering_quality(embeddings_before, labels, "before_")
    metrics_after = compute_clustering_quality(embeddings_after, labels, "after_")
    
    rows = []
    
    # Global metrics
    rows.append({
        "scope": "Global",
        "metric": "Silhouette Score",
        "before_lora": metrics_before["before_silhouette_score"],
        "after_lora": metrics_after["after_silhouette_score"],
        "change": (
            round(metrics_after["after_silhouette_score"] - metrics_before["before_silhouette_score"], 4)
            if metrics_before["before_silhouette_score"] is not None and 
               metrics_after["after_silhouette_score"] is not None
            else None
        ),
        "interpretation": "Higher is better (max=1)"
    })
    
    rows.append({
        "scope": "Global",
        "metric": "Davies-Bouldin Index",
        "before_lora": metrics_before["before_davies_bouldin_index"],
        "after_lora": metrics_after["after_davies_bouldin_index"],
        "change": (
            round(metrics_after["after_davies_bouldin_index"] - metrics_before["before_davies_bouldin_index"], 4)
            if metrics_before["before_davies_bouldin_index"] is not None and 
               metrics_after["after_davies_bouldin_index"] is not None
            else None
        ),
        "interpretation": "Lower is better (min=0)"
    })
    
    return pd.DataFrame(rows)


def compute_per_class_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """Hitung Silhouette Score per sampel, lalu rata-rata per kelas.
    
    Args:
        embeddings: Embedding array.
        labels: Label array.
        class_names: Nama kelas.
        
    Returns:
        Dict {class_name: mean_silhouette}.
    """
    from sklearn.metrics import silhouette_samples
    
    sample_scores = silhouette_samples(embeddings, labels)
    
    per_class = {}
    for i, cls_name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            per_class[cls_name] = round(np.mean(sample_scores[mask]), 4)
    
    return per_class


def main():
    parser = argparse.ArgumentParser(description="Clustering metrics (T3)")
    parser.add_argument("--config", type=str, default="configs/analysis.yaml")
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model_mae_vit.yaml")
    set_seed(config["global"]["seed"])
    
    device = config["global"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        import torch
        device = "cpu"
    
    class_names = data_config["dataset"]["classes"]
    cluster_config = config["clustering_analysis"]
    
    # Load models & extract CLS embeddings (reuse dari cls_embedding_shift)
    from src.analysis.cls_embedding_shift import extract_cls_embeddings
    from src.models.mae_vit import load_pretrained_snapshot, load_mae_vit_backbone
    from src.models.classifier_head import CassavaClassifier, build_classifier_head
    from src.data.dataset import CassavaDataset, get_val_test_transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Load pretrained backbone
    pretrained_path = resolve_path(config["checkpoints"]["pretrained_backbone"])
    backbone_pretrained = load_pretrained_snapshot(str(pretrained_path), device)
    
    # Load LoRA model
    lora_key = f"r{args.rank}"
    lora_ckpt_path = resolve_path(config["checkpoints"]["lora"][lora_key])
    backbone_lora = load_mae_vit_backbone(
        model_name=model_config["backbone"]["model_name"],
        pretrained=False, device=device
    )
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
    
    # Extract CLS (last layer only for clustering)
    print("[clustering] Extracting CLS embeddings...")
    cls_pre, labels = extract_cls_embeddings(backbone_pretrained, test_loader, device)
    cls_lora, _ = extract_cls_embeddings(model_lora, test_loader, device)
    
    # Gunakan layer terakhir
    emb_before = cls_pre[-1]
    emb_after = cls_lora[-1]
    
    # Compute comparison
    comparison_df = compare_clustering_before_after(
        emb_before, emb_after, labels, class_names
    )
    
    # Per-class silhouette
    per_class_before = compute_per_class_silhouette(emb_before, labels, class_names)
    per_class_after = compute_per_class_silhouette(emb_after, labels, class_names)
    
    print("\n[clustering] Per-class Silhouette Score:")
    print(f"  {'Class':<12} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*44}")
    for cls in class_names:
        before = per_class_before.get(cls, 0)
        after = per_class_after.get(cls, 0)
        print(f"  {cls:<12} {before:>10.4f} {after:>10.4f} {after-before:>+10.4f}")
    
    # Save
    tables_config = cluster_config["tables"]
    output_path = resolve_path(tables_config["clustering_metrics"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    
    print(f"\n[clustering] Results saved to {output_path}")
    print(comparison_df.to_string(index=False))
    
    print("\n[clustering] Done.")


if __name__ == "__main__":
    main()
