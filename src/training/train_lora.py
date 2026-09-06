"""
train_lora.py — Fase 4: LoRA fine-tuning (backbone frozen, LoRA adapters + head trainable).

Usage:
    python -m src.training.train_lora --config configs/train_lora.yaml --rank 8
    python -m src.training.train_lora --config configs/train_lora.yaml --rank 4 --target qkvo

Mendukung:
- Grid search rank (4, 8, 16, 32)
- Dua target module config: qv (Q,V) dan qkvo (Q,K,V,O)
- Simpan bobot LoRA (A, B, ΔW) secara terpisah per layer
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.seed import set_seed
from src.utils.config import load_config, resolve_path
from src.models.mae_vit import (
    load_mae_vit_backbone, freeze_backbone,
    save_pretrained_snapshot, get_backbone_info
)
from src.models.classifier_head import CassavaClassifier, build_classifier_head
from src.models.lora_layers import (
    attach_lora_peft, attach_lora_custom,
    extract_lora_weights_peft, extract_lora_weights_custom,
    save_lora_weights
)
from src.data.dataset import create_dataloaders, get_class_weights_tensor, CassavaDataset
from src.training.trainer_utils import training_loop


def train_single_lora(
    rank: int,
    target_key: str,
    train_config: dict,
    data_config: dict,
    model_config: dict,
    lora_config: dict,
    device: str
) -> dict:
    """Train satu konfigurasi LoRA.
    
    Args:
        rank: LoRA rank.
        target_key: Key untuk target modules config ("qv" atau "qkvo").
        train_config: Training config.
        data_config: Data config.
        model_config: Model config.
        lora_config: LoRA-specific config.
        device: Device.
        
    Returns:
        Dict berisi results.
    """
    alpha = rank * lora_config.get("alpha_ratio", 2)
    target_modules_configs = lora_config["target_modules_configs"]
    target_modules = target_modules_configs[target_key]["modules"]
    implementation = lora_config.get("implementation", "peft")
    
    experiment_name = f"lora_r{rank}_{target_key}"
    print(f"\n{'='*60}")
    print(f"Training LoRA: rank={rank}, alpha={alpha}, target={target_key}")
    print(f"Implementation: {implementation}")
    print(f"Target modules: {target_modules}")
    print(f"{'='*60}")
    
    # Load data
    splits_dir = resolve_path(data_config["dataset"]["splits_dir"])
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config, str(splits_dir), use_weighted_sampler=True
    )
    
    train_dataset = CassavaDataset(
        csv_path=str(splits_dir / "train.csv"),
        class_names=data_config["dataset"]["classes"]
    )
    class_weights = get_class_weights_tensor(train_dataset, num_classes=5)
    
    # Build model
    backbone = load_mae_vit_backbone(
        model_name=model_config["backbone"]["model_name"],
        pretrained=model_config["backbone"]["pretrained"],
        device=device
    )
    
    # Simpan pretrained snapshot (jika belum ada)
    pretrained_path = resolve_path(model_config["checkpoints"]["pretrained_backbone"])
    if not pretrained_path.exists():
        save_pretrained_snapshot(backbone, str(pretrained_path))
    
    # Freeze backbone
    freeze_backbone(backbone)
    
    # Attach LoRA
    if implementation == "peft":
        backbone = attach_lora_peft(
            backbone, rank=rank, alpha=alpha,
            target_modules=target_modules,
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none")
        )
    else:
        backbone = attach_lora_custom(
            backbone, rank=rank, alpha=float(alpha),
            target_modules=target_modules,
            lora_dropout=lora_config.get("lora_dropout", 0.05)
        )
    
    # Classifier head
    head_config = model_config["classifier"].copy()
    head_config["hidden_size"] = model_config["backbone"]["architecture"]["hidden_size"]
    head = build_classifier_head(head_config)
    
    model = CassavaClassifier(backbone, head).to(device)
    
    # Parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] Total params: {total_params:,}")
    print(f"[lora] Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Update save dir for this specific run
    save_dir = str(
        resolve_path(train_config["checkpointing"]["save_dir"]) / f"lora_r{rank}_{target_key}"
    )
    run_config = train_config.copy()
    run_config["checkpointing"] = {**train_config["checkpointing"], "save_dir": save_dir}
    run_config["logging"] = {
        **train_config.get("logging", {}),
        "log_dir": str(resolve_path(train_config.get("logging", {}).get(
            "log_dir", "experiments/logs/lora"
        )) / f"lora_r{rank}_{target_key}")
    }
    
    # Train
    results = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=run_config,
        device=device,
        experiment_name=experiment_name,
        separate_backbone_lr=False,
        class_weights=class_weights
    )
    
    # Evaluate on test set
    from src.training.trainer_utils import evaluate
    criterion = torch.nn.CrossEntropyLoss()
    
    best_ckpt = torch.load(
        str(Path(save_dir) / "best_model.pt"), map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n[lora r={rank}] Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Extract & save LoRA weights terpisah
    if implementation == "peft":
        lora_weights = extract_lora_weights_peft(backbone)
    else:
        lora_weights = extract_lora_weights_custom(backbone)
    
    lora_weights_path = str(Path(save_dir) / "lora_weights.pt")
    save_lora_weights(lora_weights, lora_weights_path)
    
    return {
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_key,
        "implementation": implementation,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_params / total_params * 100, 4),
        "training_time_s": results["total_time_s"],
        "num_epochs_trained": results["num_epochs_trained"],
        "best_val_metric": results["best_value"],
        "test_accuracy": test_metrics.get("val_accuracy", 0),
        "test_f1_macro": test_metrics.get("val_f1_macro", 0),
        "test_f1_weighted": test_metrics.get("val_f1_weighted", 0),
        "test_precision_macro": test_metrics.get("val_precision_macro", 0),
        "test_recall_macro": test_metrics.get("val_recall_macro", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/train_lora.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_mae_vit.yaml")
    parser.add_argument("--rank", type=int, default=None,
                        help="Specific rank to train (default: run all in grid)")
    parser.add_argument("--target", type=str, default=None,
                        help="Target modules config key (e.g., 'qv' or 'qkvo')")
    args = parser.parse_args()
    
    # Load configs
    train_config = load_config(args.config)
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    lora_config = train_config["lora"]
    
    seed = train_config["experiment"]["seed"]
    set_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine ranks to train
    if args.rank is not None:
        ranks = [args.rank]
    else:
        ranks = lora_config.get("rank_grid", [8])
    
    # Determine target modules
    if args.target is not None:
        targets = [args.target]
    else:
        targets = [lora_config.get("default_target", "qv")]
    
    # Run all combinations
    all_results = []
    
    for rank in ranks:
        for target_key in targets:
            set_seed(seed)  # Reset seed per run
            result = train_single_lora(
                rank=rank,
                target_key=target_key,
                train_config=train_config,
                data_config=data_config,
                model_config=model_config,
                lora_config=lora_config,
                device=device
            )
            all_results.append(result)
    
    # Save combined results
    results_path = resolve_path(train_config["evaluation"]["results_file"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print("LoRA Grid Search Complete")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
