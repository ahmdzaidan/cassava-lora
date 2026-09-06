"""
plotting.py — Fungsi plot terpusat untuk semua visualisasi.

Style konsisten, save ke PNG (300 dpi) + SVG.
Semua figure disimpan ke reports/figures/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ============================================================
# Style Configuration
# ============================================================

# Palet warna konsisten untuk 5 kelas penyakit singkong
CLASS_COLORS = {
    "cbb": "#E63946",      # Merah — Cassava Bacterial Blight
    "cbsd": "#457B9D",     # Biru — Cassava Brown Streak Disease
    "cgm": "#2A9D8F",      # Teal — Cassava Green Mottle
    "cmd": "#E9C46A",      # Kuning — Cassava Mosaic Disease
    "healthy": "#264653",  # Hijau tua — Healthy
}

CLASS_LABELS_PRETTY = {
    "cbb": "CBB",
    "cbsd": "CBSD",
    "cgm": "CGM",
    "cmd": "CMD",
    "healthy": "Healthy",
}

MODEL_COLORS = {
    "Linear Probe": "#A8DADC",
    "Full Fine-tune": "#457B9D",
    "LoRA": "#E63946",
}


def setup_style():
    """Set up matplotlib style konsisten untuk semua plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_figure(fig, save_path: str, formats: List[str] = None):
    """Simpan figure dalam multiple formats.
    
    Args:
        fig: Matplotlib figure.
        save_path: Base path (tanpa extension).
        formats: List format file (default: ['png', 'svg']).
    """
    if formats is None:
        formats = ["png", "svg"]
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        full_path = save_path.with_suffix(f".{fmt}")
        fig.savefig(str(full_path), format=fmt, dpi=300, bbox_inches="tight")
    
    plt.close(fig)
    print(f"[plotting] Saved: {save_path.stem} ({', '.join(formats)})")


# ============================================================
# Tujuan 1: Performa Klasifikasi
# ============================================================

def plot_accuracy_vs_params(results: List[Dict], save_path: str):
    """Plot trade-off accuracy vs jumlah parameter trainable.
    
    Args:
        results: List of result dicts dengan keys: model, params_millions, accuracy.
        save_path: Path untuk menyimpan.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for r in results:
        color = "#E63946" if "LoRA" in r.get("model", "") else \
                "#457B9D" if "Full" in r.get("model", "") else "#A8DADC"
        marker = "D" if "LoRA" in r.get("model", "") else "o"
        
        ax.scatter(
            r.get("params_millions", 0),
            r.get("accuracy", 0) * 100,
            s=150, c=color, marker=marker, edgecolors="white", linewidth=1.5,
            zorder=5, label=r.get("model", "Unknown")
        )
        
        ax.annotate(
            r.get("model", ""),
            (r.get("params_millions", 0), r.get("accuracy", 0) * 100),
            textcoords="offset points", xytext=(10, 5),
            fontsize=9, alpha=0.8
        )
    
    ax.set_xlabel("Trainable Parameters (Millions)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Parameter Efficiency: Accuracy vs Trainable Parameters")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, save_path)


def plot_f1_comparison(results: List[Dict], save_path: str):
    """Plot perbandingan F1 score antar model.
    
    Args:
        results: List of result dicts.
        save_path: Path output.
    """
    setup_style()
    
    models = [r.get("model", "") for r in results]
    f1_macro = [r.get("f1_macro", 0) * 100 for r in results]
    f1_weighted = [r.get("val_f1_weighted", r.get("f1_weighted", 0)) * 100 for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, f1_macro, width, label="Macro F1", color="#457B9D", alpha=0.9)
    bars2 = ax.bar(x + width/2, f1_weighted, width, label="Weighted F1", color="#E9C46A", alpha=0.9)
    
    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("F1 Score Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    save_figure(fig, save_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          title: str = "", save_path: Optional[str] = None,
                          ax: Optional[plt.Axes] = None):
    """Plot confusion matrix sebagai heatmap.
    
    Args:
        cm: Confusion matrix array.
        class_names: Nama kelas.
        title: Judul plot.
        save_path: Path output (opsional jika ax disediakan).
        ax: Matplotlib Axes (opsional).
    """
    setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    pretty_names = [CLASS_LABELS_PRETTY.get(n, n) for n in class_names]
    
    sns.heatmap(
        cm, annot=True, fmt=".2f" if cm.max() <= 1 else "d",
        xticklabels=pretty_names, yticklabels=pretty_names,
        cmap="Blues", ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    
    if save_path and ax is None:
        save_figure(fig, save_path)


def plot_confusion_matrices_grid(
    confusion_matrices: List[Tuple[np.ndarray, str]],
    class_names: List[str],
    save_path: str
):
    """Plot multiple confusion matrices dalam grid.
    
    Args:
        confusion_matrices: List of (cm_array, model_name) tuples.
        class_names: Nama kelas.
        save_path: Path output.
    """
    setup_style()
    n = len(confusion_matrices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    
    if n == 1:
        axes = [axes]
    
    for ax, (cm, title) in zip(axes, confusion_matrices):
        plot_confusion_matrix(cm, class_names, title=title, ax=ax)
    
    fig.suptitle("Confusion Matrices Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path)


# ============================================================
# Tujuan 2: SVD & Attention
# ============================================================

def plot_svd_decay(svd_results: Dict, save_path: str):
    """Plot kurva peluruhan nilai singular per layer.
    
    Args:
        svd_results: Dict {layer_name: svd_analysis_results}.
        save_path: Path output.
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    cmap = plt.cm.viridis
    n_layers = len(svd_results)
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]
    
    # Plot 1: Singular values (raw)
    for i, (layer_name, result) in enumerate(svd_results.items()):
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        S = result["S"]
        axes[0].plot(range(len(S)), S, marker=".", markersize=3,
                     color=colors[i], label=f"L{i}", alpha=0.8)
    
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title("Singular Value Decay of ΔW = B·A per Layer")
    axes[0].set_yscale("log")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative energy
    for i, (layer_name, result) in enumerate(svd_results.items()):
        energy = result["cumulative_energy"]
        axes[1].plot(range(len(energy)), energy * 100,
                     color=colors[i], label=f"L{i}", alpha=0.8)
    
    axes[1].axhline(y=99, color="red", linestyle="--", alpha=0.5, label="99% energy")
    axes[1].set_xlabel("Number of Singular Values")
    axes[1].set_ylabel("Cumulative Energy (%)")
    axes[1].set_title("Cumulative Energy of ΔW Singular Values")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_intruder_dimensions(intruder_results: Dict, save_path: str):
    """Plot deteksi intruder dimensions.
    
    Args:
        intruder_results: Dict {layer_name: intruder_detection_results}.
        save_path: Path output.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    layers = list(intruder_results.keys())
    
    # Plot 1: Cosine similarity distribution
    for i, (layer_name, result) in enumerate(intruder_results.items()):
        cos_sims = result["cosine_similarities"]
        short_name = f"L{i}"
        axes[0].scatter(
            [i] * len(cos_sims), cos_sims,
            alpha=0.6, s=20, label=short_name if i < 5 else ""
        )
    
    threshold = list(intruder_results.values())[0]["threshold"] if intruder_results else 0.1
    axes[0].axhline(y=threshold, color="red", linestyle="--",
                    alpha=0.7, label=f"Threshold={threshold}")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Cosine Similarity with W₀ Subspace")
    axes[0].set_title("Intruder Dimension Detection")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Number of intruders per layer
    num_intruders = [result["num_intruders"] for result in intruder_results.values()]
    layer_indices = range(len(num_intruders))
    
    axes[1].bar(layer_indices, num_intruders, color="#E63946", alpha=0.8)
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Number of Intruder Dimensions")
    axes[1].set_title("Intruder Dimensions per Layer")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_attention_entropy_comparison(
    entropy_before: Dict,
    entropy_after: Dict,
    save_path: str
):
    """Plot perbandingan entropi attention sebelum vs sesudah LoRA.
    
    Args:
        entropy_before: Entropy dict dari model pretrained.
        entropy_after: Entropy dict dari model LoRA.
        save_path: Path output.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = range(len(entropy_before["layer_entropy"]))
    
    ax.plot(layers, entropy_before["layer_entropy"], "o-",
            color="#457B9D", label="Before LoRA (Pretrained)", linewidth=2)
    ax.plot(layers, entropy_after["layer_entropy"], "s-",
            color="#E63946", label="After LoRA", linewidth=2)
    
    ax.fill_between(layers,
                     entropy_before["layer_entropy"],
                     entropy_after["layer_entropy"],
                     alpha=0.1, color="#E9C46A")
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Attention Entropy (CLS token)")
    ax.set_title("Attention Entropy: Before vs After LoRA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, save_path)


def plot_attention_heatmap_overlay(
    image: Any,
    heatmap_before: np.ndarray,
    heatmap_after: np.ndarray,
    save_path: str
):
    """Plot attention heatmap overlay pada citra.
    
    Args:
        image: Image tensor (C, H, W) atau PIL Image.
        heatmap_before: Attention heatmap pretrained.
        heatmap_after: Attention heatmap LoRA.
        save_path: Path output.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Denormalize image
    import torch
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).numpy()
        # Denormalize ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
    else:
        img = np.array(image) / 255.0
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Before LoRA
    axes[1].imshow(img)
    import matplotlib.cm as cm
    hm_resized = np.kron(heatmap_before, np.ones((16, 16)))[:img.shape[0], :img.shape[1]]
    axes[1].imshow(hm_resized, cmap="jet", alpha=0.4)
    axes[1].set_title("Attention: Before LoRA")
    axes[1].axis("off")
    
    # After LoRA
    axes[2].imshow(img)
    hm_resized = np.kron(heatmap_after, np.ones((16, 16)))[:img.shape[0], :img.shape[1]]
    axes[2].imshow(hm_resized, cmap="jet", alpha=0.4)
    axes[2].set_title("Attention: After LoRA")
    axes[2].axis("off")
    
    plt.suptitle("Attention Map Comparison", fontsize=14)
    plt.tight_layout()
    save_figure(fig, save_path)


# ============================================================
# Tujuan 3: CLS Shift & Clustering
# ============================================================

def plot_cls_shift_per_layer(shift_results: Dict, save_path: str):
    """Plot rata-rata cosine shift per layer.
    
    Args:
        shift_results: Output dari compute_cosine_shift().
        save_path: Path output.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_shifts = shift_results["mean_shift_per_layer"]
    std_shifts = shift_results["std_shift_per_layer"]
    layers = range(len(mean_shifts))
    
    ax.plot(layers, mean_shifts, "o-", color="#E63946", linewidth=2.5,
            markersize=8, label="Mean Shift")
    ax.fill_between(layers,
                     mean_shifts - std_shifts,
                     mean_shifts + std_shifts,
                     alpha=0.2, color="#E63946", label="±1 Std Dev")
    
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Shift (1 - cos(CLS_pre, CLS_LoRA))")
    ax.set_title("[CLS] Token Cosine Shift per Layer\n(Higher = More Change)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotate layer 0 (embedding) dan layer terakhir
    ax.annotate(f"{mean_shifts[0]:.4f}", (0, mean_shifts[0]),
                textcoords="offset points", xytext=(15, 10),
                fontsize=9, color="#457B9D")
    ax.annotate(f"{mean_shifts[-1]:.4f}", (len(mean_shifts)-1, mean_shifts[-1]),
                textcoords="offset points", xytext=(-30, -20),
                fontsize=9, color="#457B9D")
    
    save_figure(fig, save_path)


def plot_embedding_scatter(
    reduced_before: np.ndarray,
    reduced_after: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title_before: str = "Before LoRA",
    title_after: str = "After LoRA",
    method_name: str = "t-SNE",
    save_path: str = ""
):
    """Plot scatter 2D sebelum/sesudah LoRA side by side.
    
    Args:
        reduced_before: 2D coords before LoRA, shape (n, 2).
        reduced_after: 2D coords after LoRA, shape (n, 2).
        labels: Class labels.
        class_names: Nama kelas.
        title_before: Judul panel kiri.
        title_after: Judul panel kanan.
        method_name: Nama metode (t-SNE / UMAP).
        save_path: Path output.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    color_list = [CLASS_COLORS.get(cn, "#333333") for cn in class_names]
    
    for ax, reduced, title in [
        (axes[0], reduced_before, title_before),
        (axes[1], reduced_after, title_after)
    ]:
        for i, cls_name in enumerate(class_names):
            mask = labels == i
            pretty = CLASS_LABELS_PRETTY.get(cls_name, cls_name)
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=color_list[i], s=15, alpha=0.6,
                label=pretty, edgecolors="none"
            )
        
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f"{method_name} Dimension 1")
        ax.set_ylabel(f"{method_name} Dimension 2")
        ax.legend(markerscale=3, fontsize=9)
        ax.grid(True, alpha=0.2)
    
    fig.suptitle(f"{method_name} Visualization of [CLS] Embeddings", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_cka_heatmap(cka_matrix: np.ndarray, layer_names: List[str], save_path: str):
    """Plot CKA similarity heatmap.
    
    Args:
        cka_matrix: CKA matrix of shape (n_layers, n_layers).
        layer_names: Nama layer.
        save_path: Path output.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cka_matrix, annot=True, fmt=".2f",
        xticklabels=[f"L{i}" for i in range(len(layer_names))],
        yticklabels=[f"L{i}" for i in range(len(layer_names))],
        cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
        cbar_kws={"label": "Linear CKA"}
    )
    ax.set_xlabel("LoRA Model Layer")
    ax.set_ylabel("Pretrained Model Layer")
    ax.set_title("CKA Similarity: Pretrained vs LoRA-adapted Layers")
    
    save_figure(fig, save_path)
