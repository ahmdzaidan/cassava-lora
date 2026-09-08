"""
dimensionality_reduction.py — t-SNE/UMAP visualisasi [CLS] embedding.

Visualisasi 2D dari [CLS] embedding sebelum vs sesudah LoRA,
diwarnai per kelas penyakit.

Usage:
    python -m src.analysis.dimensionality_reduction --config configs/analysis.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path


def run_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    n_components: int = 2
) -> np.ndarray:
    """Jalankan t-SNE dimensionality reduction.
    
    Args:
        embeddings: Input features of shape (n_samples, n_features).
        perplexity: t-SNE perplexity parameter.
        n_iter: Jumlah iterasi.
        random_state: Random seed.
        n_components: Dimensi output (biasanya 2).
        
    Returns:
        Reduced embeddings of shape (n_samples, n_components).
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )
    
    reduced = tsne.fit_transform(embeddings)
    print(f"[dimred] t-SNE complete: {embeddings.shape} → {reduced.shape}, "
          f"KL divergence: {tsne.kl_divergence_:.4f}")
    
    return reduced


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    n_components: int = 2
) -> np.ndarray:
    """Jalankan UMAP dimensionality reduction.
    
    Args:
        embeddings: Input features of shape (n_samples, n_features).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: Distance metric.
        random_state: Random seed.
        n_components: Dimensi output.
        
    Returns:
        Reduced embeddings of shape (n_samples, n_components).
    """
    try:
        import umap
    except ImportError:
        print("[dimred] ERROR: umap-learn not installed. Run: pip install umap-learn")
        return None
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    reduced = reducer.fit_transform(embeddings)
    print(f"[dimred] UMAP complete: {embeddings.shape} → {reduced.shape}")
    
    return reduced


def run_reduction_before_after(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Jalankan dimensionality reduction untuk sebelum dan sesudah LoRA.
    
    Penting: kedua set embedding di-fit bersama untuk perbandingan yang adil.
    
    Args:
        embeddings_before: CLS sebelum LoRA.
        embeddings_after: CLS sesudah LoRA.
        labels: Class labels.
        method: 'tsne' atau 'umap'.
        **kwargs: Parameter tambahan untuk method.
        
    Returns:
        Tuple (reduced_before, reduced_after).
    """
    n_before = len(embeddings_before)
    
    # Gabungkan untuk fit bersama
    combined = np.vstack([embeddings_before, embeddings_after])
    
    if method == "tsne":
        reduced = run_tsne(combined, **kwargs)
    elif method == "umap":
        reduced = run_umap(combined, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if reduced is None:
        return None, None
    
    reduced_before = reduced[:n_before]
    reduced_after = reduced[n_before:]
    
    return reduced_before, reduced_after


def main():
    parser = argparse.ArgumentParser(description="Dimensionality reduction (T3)")
    parser.add_argument("--config", type=str, default="configs/analysis.yaml")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--method", type=str, default="both",
                        choices=["tsne", "umap", "both"])
    args = parser.parse_args()
    
    config = load_config(args.config)
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model_mae_vit.yaml")
    set_seed(config["global"]["seed"])
    
    import torch
    device = config["global"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    class_names = data_config["dataset"]["classes"]
    dimred_config = config["dimensionality_reduction"]
    
    # Load models & extract CLS
    from src.analysis.cls_embedding_shift import extract_cls_embeddings
    from src.models.mae_vit import load_pretrained_snapshot, load_mae_vit_backbone
    from src.models.classifier_head import CassavaClassifier, build_classifier_head
    from src.data.dataset import CassavaDataset, get_val_test_transforms
    from torch.utils.data import DataLoader
    from src.models.lora_layers import rebuild_lora_backbone

    
    pretrained_path = resolve_path(config["checkpoints"]["pretrained_backbone"])
    backbone_pretrained = load_pretrained_snapshot(str(pretrained_path), device)
    
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
    
    splits_dir = resolve_path(data_config["dataset"]["splits_dir"])
    transform = get_val_test_transforms(data_config)
    test_dataset = CassavaDataset(
        str(splits_dir / "test.csv"), transform=transform,
        class_names=class_names
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print("[dimred] Extracting CLS embeddings...")
    cls_pre, labels = extract_cls_embeddings(backbone_pretrained, test_loader, device)
    cls_lora, _ = extract_cls_embeddings(model_lora, test_loader, device)
    
    # Last layer embeddings
    emb_before = cls_pre[-1]
    emb_after = cls_lora[-1]
    
    figures_config = dimred_config["figures"]
    
    # t-SNE
    if args.method in ["tsne", "both"]:
        print("\n[dimred] Running t-SNE...")
        tsne_config = dimred_config["tsne"]
        reduced_before, reduced_after = run_reduction_before_after(
            emb_before, emb_after, labels, method="tsne",
            perplexity=tsne_config["perplexity"],
            n_iter=tsne_config["n_iter"],
            random_state=tsne_config["random_state"]
        )
        
        if reduced_before is not None:
            try:
                from src.viz.plotting import plot_embedding_scatter
                
                plot_embedding_scatter(
                    reduced_before, reduced_after, labels, class_names,
                    title_before="Before LoRA (Pretrained)",
                    title_after="After LoRA",
                    method_name="t-SNE",
                    save_path=str(resolve_path(figures_config["tsne_before_after"]))
                )
                print("[dimred] t-SNE figures saved.")
            except ImportError as e:
                print(f"[dimred] WARNING: {e}")
    
    # UMAP
    if args.method in ["umap", "both"]:
        print("\n[dimred] Running UMAP...")
        umap_config = dimred_config["umap"]
        reduced_before, reduced_after = run_reduction_before_after(
            emb_before, emb_after, labels, method="umap",
            n_neighbors=umap_config["n_neighbors"],
            min_dist=umap_config["min_dist"],
            metric=umap_config["metric"],
            random_state=umap_config["random_state"]
        )
        
        if reduced_before is not None:
            try:
                from src.viz.plotting import plot_embedding_scatter
                
                plot_embedding_scatter(
                    reduced_before, reduced_after, labels, class_names,
                    title_before="Before LoRA (Pretrained)",
                    title_after="After LoRA",
                    method_name="UMAP",
                    save_path=str(resolve_path(figures_config["umap_before_after"]))
                )
                print("[dimred] UMAP figures saved.")
            except ImportError as e:
                print(f"[dimred] WARNING: {e}")
    
    print("\n[dimred] Done.")


if __name__ == "__main__":
    main()
