"""
evaluate.py — Fase 5: Evaluasi komparatif semua model (Tujuan 1).

Usage:
    python -m src.evaluation.evaluate --config configs/analysis.yaml --objective T1

Mengumpulkan hasil dari semua varian model (linear probe, full fine-tune, LoRA)
dan menghasilkan tabel komparatif serta visualisasi untuk T1.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    get_classification_report_str,
    compute_efficiency_metrics
)
from src.data.dataset import CassavaDataset, get_val_test_transforms, create_dataloaders
from src.models.mae_vit import load_mae_vit_backbone, freeze_backbone
from src.models.classifier_head import CassavaClassifier, build_classifier_head


def evaluate_model_on_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
    class_names: list
) -> dict:
    """Evaluasi satu model pada test set.
    
    Args:
        model: Model yang sudah di-load checkpoint-nya.
        test_loader: Test DataLoader.
        device: Device.
        class_names: Nama kelas.
        
    Returns:
        Dict berisi semua metrik + confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(pixel_values=images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrik
    metrics = compute_classification_metrics(all_preds=all_preds, y_true=all_labels, y_pred=all_preds, class_names=class_names)
    cm, _ = compute_confusion_matrix(all_labels, all_preds, class_names)
    report = get_classification_report_str(all_labels, all_preds, class_names)
    
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
    }


def compile_results_from_files() -> list:
    """Kumpulkan hasil dari file JSON/CSV yang sudah disimpan oleh training scripts.
    
    Returns:
        List of result dicts.
    """
    results = []
    
    # Linear probe
    lp_path = resolve_path("reports/tables/results_linear_probe.json")
    if lp_path.exists():
        with open(lp_path) as f:
            lp = json.load(f)
            results.append({
                "model": "Linear Probe",
                "trainable_params": lp["trainable_params"],
                "total_params": lp["total_params"],
                "trainable_pct": lp["trainable_pct"],
                "training_time_s": lp["training_time_s"],
                **{k: v for k, v in lp.get("test_metrics", {}).items()}
            })
    
    # Full fine-tune
    ff_path = resolve_path("reports/tables/results_full_finetune.json")
    if ff_path.exists():
        with open(ff_path) as f:
            ff = json.load(f)
            results.append({
                "model": "Full Fine-tune",
                "trainable_params": ff["trainable_params"],
                "total_params": ff["total_params"],
                "trainable_pct": ff["trainable_pct"],
                "training_time_s": ff["training_time_s"],
                **{k: v for k, v in ff.get("test_metrics", {}).items()}
            })
    
    # LoRA grid results
    lora_path = resolve_path("reports/tables/results_lora_grid.csv")
    if lora_path.exists():
        lora_df = pd.read_csv(lora_path)
        for _, row in lora_df.iterrows():
            results.append({
                "model": f"LoRA r={int(row['rank'])} ({row.get('target_modules', 'qv')})",
                "trainable_params": int(row["trainable_params"]),
                "total_params": int(row["total_params"]),
                "trainable_pct": row["trainable_pct"],
                "training_time_s": row.get("training_time_s", 0),
                "val_accuracy": row.get("test_accuracy", 0),
                "val_f1_macro": row.get("test_f1_macro", 0),
                "val_f1_weighted": row.get("test_f1_weighted", 0),
                "val_precision_macro": row.get("test_precision_macro", 0),
                "val_recall_macro": row.get("test_recall_macro", 0),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare all models (T1)")
    parser.add_argument("--config", type=str, default="configs/analysis.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--objective", type=str, default="T1")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["global"]["seed"])
    
    # Compile results
    results = compile_results_from_files()
    
    if not results:
        print("[evaluate.py] Tidak ada hasil training yang ditemukan.")
        print("  Jalankan training scripts (Fase 2-4) terlebih dahulu.")
        return
    
    # Add efficiency metrics
    for r in results:
        r["accuracy"] = r.get("val_accuracy", 0)
        r["f1_macro"] = r.get("val_f1_macro", 0)
    
    enriched_results = compute_efficiency_metrics(results)
    
    # Save comparison table
    t1_config = config["evaluation_t1"]
    output_path = resolve_path(t1_config["output_table"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(enriched_results)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("COMPARISON TABLE — Tujuan 1 (Performa Klasifikasi)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nSaved to {output_path}")
    
    # Generate figures (delegated to plotting module)
    try:
        from src.viz.plotting import (
            plot_accuracy_vs_params,
            plot_confusion_matrices_grid,
            plot_f1_comparison
        )
        
        figures_config = t1_config["figures"]
        
        plot_accuracy_vs_params(
            enriched_results,
            save_path=str(resolve_path(figures_config["accuracy_vs_params"]))
        )
        
        plot_f1_comparison(
            enriched_results,
            save_path=str(resolve_path(figures_config["f1_comparison"]))
        )
        
        print("[evaluate.py] Figures generated successfully.")
    except ImportError as e:
        print(f"[evaluate.py] WARNING: Could not generate figures: {e}")
    
    print("\n[evaluate.py] Done.")


if __name__ == "__main__":
    main()
