"""
train_full_finetune.py — Fase 3: Full fine-tuning (semua parameter trainable).

Usage:
    python -m src.training.train_full_finetune --config configs/train_full_finetune.yaml

Seluruh backbone + head dilatih end-to-end. Ini adalah upper bound performa
dan baseline atas untuk perbandingan efisiensi vs LoRA.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path
from src.models.mae_vit import (
    load_mae_vit_backbone, unfreeze_backbone, 
    save_pretrained_snapshot, get_backbone_info
)
from src.models.classifier_head import CassavaClassifier, build_classifier_head
from src.data.dataset import create_dataloaders, get_class_weights_tensor, CassavaDataset
from src.training.trainer_utils import training_loop


def main():
    parser = argparse.ArgumentParser(description="Train full fine-tuning")
    parser.add_argument("--config", type=str, default="configs/train_full_finetune.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_mae_vit.yaml")
    args = parser.parse_args()
    
    # Load configs
    train_config = load_config(args.config)
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    
    seed = train_config["experiment"]["seed"]
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[full_finetune] Device: {device}")
    
    # ==========================================
    # 1. Load Data
    # ==========================================
    splits_dir = resolve_path(data_config["dataset"]["splits_dir"])
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config, str(splits_dir), use_weighted_sampler=True
    )
    
    # Class weights
    train_dataset = CassavaDataset(
        csv_path=str(splits_dir / "train.csv"),
        class_names=data_config["dataset"]["classes"]
    )
    class_weights = get_class_weights_tensor(train_dataset, num_classes=5)
    
    # ==========================================
    # 2. Build Model
    # ==========================================
    backbone = load_mae_vit_backbone(
        model_name=model_config["backbone"]["model_name"],
        pretrained=model_config["backbone"]["pretrained"],
        device=device
    )
    
    # Simpan snapshot pretrained (jika belum disimpan oleh linear probe)
    pretrained_path = resolve_path(model_config["checkpoints"]["pretrained_backbone"])
    if not pretrained_path.exists():
        save_pretrained_snapshot(backbone, str(pretrained_path))
    
    # Unfreeze backbone (full fine-tuning)
    unfreeze_backbone(backbone)
    
    # Build classifier head
    head_config = model_config["classifier"].copy()
    head_config["hidden_size"] = model_config["backbone"]["architecture"]["hidden_size"]
    head = build_classifier_head(head_config)
    
    model = CassavaClassifier(backbone, head).to(device)
    
    # Parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[full_finetune] Total params: {total_params:,}")
    print(f"[full_finetune] Trainable params: {trainable_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")
    
    # ==========================================
    # 3. Train
    # ==========================================
    results = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        experiment_name="full_finetune",
        separate_backbone_lr=True,  # Discriminative LR
        class_weights=class_weights
    )
    
    # ==========================================
    # 4. Evaluate on Test Set
    # ==========================================
    from src.training.trainer_utils import evaluate
    criterion = torch.nn.CrossEntropyLoss()
    
    best_ckpt = torch.load(
        str(resolve_path(train_config["checkpointing"]["save_dir"]) / "best_model.pt"),
        map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n[full_finetune] Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # ==========================================
    # 5. Save Results
    # ==========================================
    results_path = resolve_path(train_config["evaluation"]["results_file"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "experiment": "full_finetune",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_params / total_params * 100, 4),
        "training_time_s": results["total_time_s"],
        "num_epochs_trained": results["num_epochs_trained"],
        "best_val_metric": results["best_value"],
        "test_metrics": test_metrics,
    }
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n[full_finetune] Results saved to {results_path}")
    print("[full_finetune] Done.")


if __name__ == "__main__":
    main()
