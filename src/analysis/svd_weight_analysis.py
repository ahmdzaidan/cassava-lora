"""
svd_weight_analysis.py — Fase 6, Tujuan 2: Analisis SVD pada bobot LoRA.

Analisis:
1. SVD pada ΔW = B·A per layer attention
2. Plot kurva peluruhan nilai singular per layer
3. Deteksi intruder dimensions
4. Perbandingan basis singular ΔW vs W0 (pretrained)

Usage:
    python -m src.analysis.svd_weight_analysis --config configs/analysis.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path
from src.models.lora_layers import load_lora_weights


def svd_analysis_single_layer(
    delta_w: torch.Tensor
) -> Dict[str, np.ndarray]:
    """Lakukan SVD pada ΔW satu layer.
    
    Args:
        delta_w: Tensor ΔW = B·A of shape (out_features, in_features).
        
    Returns:
        Dict berisi U, S, Vh, dan informasi terkait.
    """
    if isinstance(delta_w, torch.Tensor):
        delta_w = delta_w.detach().cpu().numpy()
    
    U, S, Vh = np.linalg.svd(delta_w, full_matrices=False)
    
    # Normalized singular values (relatif terhadap yang terbesar)
    S_normalized = S / (S[0] + 1e-10)
    
    # Cumulative energy
    energy = np.cumsum(S**2) / (np.sum(S**2) + 1e-10)
    
    # Effective rank (jumlah SV yang dibutuhkan untuk 99% energy)
    effective_rank = np.searchsorted(energy, 0.99) + 1
    
    return {
        "U": U,
        "S": S,
        "Vh": Vh,
        "S_normalized": S_normalized,
        "cumulative_energy": energy,
        "effective_rank": effective_rank,
        "frobenius_norm": np.linalg.norm(delta_w, "fro"),
    }


def analyze_all_layers(
    lora_weights: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Lakukan SVD analysis untuk semua layers.
    
    Args:
        lora_weights: Dict LoRA weights dari load_lora_weights().
        
    Returns:
        Dict {layer_name: svd_results}.
    """
    results = {}
    
    for layer_name, weights in lora_weights.items():
        if "delta_W" not in weights:
            # Hitung ΔW dari A dan B
            A = weights["A"]
            B = weights["B"]
            if isinstance(A, torch.Tensor):
                delta_w = B @ A
            else:
                delta_w = torch.tensor(B) @ torch.tensor(A)
        else:
            delta_w = weights["delta_W"]
        
        results[layer_name] = svd_analysis_single_layer(delta_w)
        print(f"  SVD [{layer_name}]: "
              f"rank={results[layer_name]['effective_rank']}, "
              f"‖ΔW‖_F={results[layer_name]['frobenius_norm']:.6f}")
    
    return results


def detect_intruder_dimensions(
    delta_w_svd: Dict[str, np.ndarray],
    w0: np.ndarray,
    top_k: int = 10,
    threshold: float = 0.1
) -> Dict[str, any]:
    """Deteksi intruder dimensions pada ΔW relatif terhadap W0.
    
    Vektor singular dari ΔW yang hampir ortogonal terhadap
    subruang top-k dari W0 ditandai sebagai intruder dimension.
    
    Args:
        delta_w_svd: SVD results dari svd_analysis_single_layer().
        w0: Bobot pretrained asli (numpy array).
        top_k: Jumlah top singular vectors W0 untuk perbandingan.
        threshold: Cosine similarity di bawah ini = intruder.
        
    Returns:
        Dict berisi informasi intruder dimensions.
    """
    # SVD pada W0
    U0, S0, Vh0 = np.linalg.svd(w0, full_matrices=False)
    
    # Top-k right singular vectors dari W0
    Vh0_top = Vh0[:top_k, :]
    
    # Right singular vectors dari ΔW
    Vh_delta = delta_w_svd["Vh"]
    
    # Hitung principal angles antara setiap SV dari ΔW dan subruang W0
    intruder_indices = []
    cosine_similarities = []
    
    for i in range(min(len(Vh_delta), top_k)):
        v_delta = Vh_delta[i:i+1, :]  # shape (1, d)
        
        # Proyeksi v_delta ke subruang Vh0_top
        projection = v_delta @ Vh0_top.T  # shape (1, top_k)
        cos_sim = np.max(np.abs(projection))
        cosine_similarities.append(cos_sim)
        
        if cos_sim < threshold:
            intruder_indices.append(i)
    
    # Subspace angles
    try:
        min_dim = min(Vh_delta.shape[0], Vh0_top.shape[0], Vh_delta.shape[1])
        angles = subspace_angles(Vh_delta[:min_dim].T, Vh0_top[:min_dim].T)
        angles_degrees = np.degrees(angles)
    except Exception:
        angles_degrees = np.array([])
    
    return {
        "intruder_indices": intruder_indices,
        "num_intruders": len(intruder_indices),
        "cosine_similarities": np.array(cosine_similarities),
        "subspace_angles_degrees": angles_degrees,
        "threshold": threshold,
        "top_k": top_k,
    }


def run_full_svd_analysis(
    lora_weights_path: str,
    pretrained_state_dict: dict,
    top_k: int = 10,
    intruder_threshold: float = 0.1
) -> Tuple[Dict, pd.DataFrame]:
    """Jalankan analisis SVD lengkap dan generate summary.
    
    Args:
        lora_weights_path: Path ke file lora_weights.pt.
        pretrained_state_dict: State dict backbone pretrained.
        top_k: Top-k singular vectors untuk perbandingan.
        intruder_threshold: Threshold deteksi intruder.
        
    Returns:
        Tuple (full_results_dict, summary_dataframe).
    """
    print("[svd_analysis] Loading LoRA weights...")
    lora_weights = load_lora_weights(lora_weights_path)
    
    print("[svd_analysis] Running SVD analysis per layer...")
    svd_results = analyze_all_layers(lora_weights)
    
    # Intruder detection per layer
    print("[svd_analysis] Detecting intruder dimensions...")
    intruder_results = {}
    
    for layer_name in svd_results:
        # Cari bobot W0 yang sesuai di pretrained state dict
        # Mapping nama layer LoRA ke nama parameter pretrained
        w0_key = None
        for key in pretrained_state_dict:
            # Coba match berdasarkan pattern
            if any(part in key for part in layer_name.split(".")):
                if "weight" in key and pretrained_state_dict[key].dim() == 2:
                    w0_key = key
                    break
        
        if w0_key is not None:
            w0 = pretrained_state_dict[w0_key]
            if isinstance(w0, torch.Tensor):
                w0 = w0.detach().cpu().numpy()
            
            intruder_results[layer_name] = detect_intruder_dimensions(
                svd_results[layer_name], w0,
                top_k=top_k, threshold=intruder_threshold
            )
        else:
            print(f"  WARNING: W0 not found for {layer_name}")
    
    # Build summary DataFrame
    rows = []
    for layer_name in svd_results:
        svd = svd_results[layer_name]
        row = {
            "layer": layer_name,
            "effective_rank": svd["effective_rank"],
            "frobenius_norm": round(svd["frobenius_norm"], 6),
            "top_sv": round(svd["S"][0], 6) if len(svd["S"]) > 0 else 0,
            "sv_ratio_1_2": round(svd["S"][0] / (svd["S"][1] + 1e-10), 4) if len(svd["S"]) > 1 else 0,
        }
        
        if layer_name in intruder_results:
            intr = intruder_results[layer_name]
            row["num_intruders"] = intr["num_intruders"]
            row["avg_cos_sim"] = round(np.mean(intr["cosine_similarities"]), 4)
        
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    full_results = {
        "svd_results": svd_results,
        "intruder_results": intruder_results,
    }
    
    return full_results, summary_df


def main():
    parser = argparse.ArgumentParser(description="SVD analysis on LoRA weights (T2)")
    parser.add_argument("--config", type=str, default="configs/analysis.yaml")
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["global"]["seed"])
    
    svd_config = config["svd_analysis"]
    
    # Load LoRA weights
    lora_key = f"r{args.rank}"
    lora_weights_path = resolve_path(config["checkpoints"]["lora_weights"][lora_key])
    
    # Load pretrained backbone state dict
    pretrained_path = resolve_path(config["checkpoints"]["pretrained_backbone"])
    pretrained_ckpt = torch.load(str(pretrained_path), map_location="cpu")
    pretrained_state_dict = pretrained_ckpt["model_state_dict"]
    
    # Run analysis
    full_results, summary_df = run_full_svd_analysis(
        str(lora_weights_path),
        pretrained_state_dict,
        top_k=svd_config["top_k_singular"],
        intruder_threshold=svd_config["intruder_threshold"]
    )
    
    # Save summary
    summary_path = resolve_path(svd_config["tables"]["svd_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print("SVD Analysis Summary")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to {summary_path}")
    
    # Generate figures
    try:
        from src.viz.plotting import plot_svd_decay, plot_intruder_dimensions
        
        figures_config = svd_config["figures"]
        
        plot_svd_decay(
            full_results["svd_results"],
            save_path=str(resolve_path(figures_config["svd_decay"]))
        )
        
        if full_results["intruder_results"]:
            plot_intruder_dimensions(
                full_results["intruder_results"],
                save_path=str(resolve_path(figures_config["intruder_dimensions"]))
            )
        
        print("[svd_analysis] Figures generated.")
    except ImportError as e:
        print(f"[svd_analysis] WARNING: Could not generate figures: {e}")
    
    print("\n[svd_analysis] Done.")


if __name__ == "__main__":
    main()
