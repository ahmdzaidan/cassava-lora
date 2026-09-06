"""
trainer_utils.py — Generic training utilities.

Fitur:
- Training loop dengan mixed precision (AMP)
- Early stopping
- Checkpointing (best + last)
- Learning rate scheduler
- CSV & TensorBoard logging
- Gradient clipping & accumulation
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Early stopping untuk mencegah overfitting.
    
    Args:
        patience: Jumlah epoch tanpa improvement sebelum berhenti.
        mode: 'min' (loss) atau 'max' (accuracy/F1).
        min_delta: Minimum improvement yang dianggap signifikan.
    """
    
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check apakah harus berhenti.
        
        Args:
            value: Nilai metrik yang di-monitor.
            
        Returns:
            True jika harus berhenti.
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CSVLogger:
    """Logger sederhana ke file CSV.
    
    Args:
        log_path: Path ke file CSV.
    """
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Tulis satu baris metrik ke CSV."""
        mode = "a" if self.header_written else "w"
        with open(self.log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(metrics)


class CheckpointManager:
    """Manage checkpoint (save best & last).
    
    Args:
        save_dir: Directory untuk checkpoint.
        monitor: Nama metrik yang di-monitor.
        mode: 'min' atau 'max'.
    """
    
    def __init__(self, save_dir: str, monitor: str = "val_f1_macro", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
    
    def save(
        self, 
        model: nn.Module, 
        optimizer: Any, 
        epoch: int, 
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Simpan checkpoint jika metrik improved.
        
        Args:
            model: Model yang akan disimpan.
            optimizer: Optimizer state.
            epoch: Epoch saat ini.
            metrics: Dict metrik.
            extra: Data tambahan (e.g., scheduler state).
            
        Returns:
            True jika best checkpoint diperbarui.
        """
        current_value = metrics.get(self.monitor, 0)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        if extra:
            checkpoint.update(extra)
        
        # Selalu simpan last
        torch.save(checkpoint, str(self.save_dir / "last_model.pt"))
        
        # Simpan best jika improved
        is_best = False
        if self.mode == "max" and current_value > self.best_value:
            is_best = True
        elif self.mode == "min" and current_value < self.best_value:
            is_best = True
        
        if is_best:
            self.best_value = current_value
            torch.save(checkpoint, str(self.save_dir / "best_model.pt"))
            print(f"  [checkpoint] New best: {self.monitor}={current_value:.4f}")
        
        return is_best


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    separate_backbone_lr: bool = False
) -> AdamW:
    """Buat optimizer dengan optional discriminative learning rate.
    
    Args:
        model: Model.
        config: Training config (bagian 'optimizer').
        separate_backbone_lr: Jika True, backbone pakai lr lebih kecil.
        
    Returns:
        AdamW optimizer.
    """
    opt_config = config["optimizer"]
    
    if separate_backbone_lr and "lr_backbone" in opt_config:
        # Discriminative LR: backbone lr kecil, head lr besar
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name or "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = [
            {"params": backbone_params, "lr": opt_config["lr_backbone"]},
            {"params": head_params, "lr": opt_config["lr_head"]},
        ]
    else:
        lr = opt_config.get("lr", opt_config.get("lr_head", 1e-3))
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}
        ]
    
    optimizer = AdamW(
        param_groups,
        weight_decay=opt_config.get("weight_decay", 0.01),
        betas=tuple(opt_config.get("betas", [0.9, 0.999]))
    )
    
    return optimizer


def create_scheduler(optimizer: Any, config: Dict[str, Any], num_epochs: int) -> Any:
    """Buat learning rate scheduler.
    
    Args:
        optimizer: Optimizer.
        config: Training config (bagian 'scheduler').
        num_epochs: Total epochs.
        
    Returns:
        LR scheduler.
    """
    sched_config = config["scheduler"]
    sched_type = sched_config.get("type", "cosine")
    
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - sched_config.get("warmup_epochs", 0),
            eta_min=sched_config.get("min_lr", 1e-7)
        )
    elif sched_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5
        )
    elif sched_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=sched_config.get("step_size", 10),
            gamma=sched_config.get("gamma", 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")
    
    return scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Any,
    criterion: nn.Module,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    log_every_n_steps: int = 10
) -> Dict[str, float]:
    """Satu epoch training.
    
    Args:
        model: Model.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device ('cuda'/'cpu').
        epoch: Current epoch number.
        scaler: GradScaler untuk mixed precision.
        max_grad_norm: Max gradient norm untuk clipping.
        gradient_accumulation_steps: Gradient accumulation steps.
        log_every_n_steps: Print log setiap N steps.
        
    Returns:
        Dict metrik training (loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
    
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision forward
        use_amp = scaler is not None
        with autocast(enabled=use_amp):
            outputs = model(pixel_values=images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            loss = loss / gradient_accumulation_steps
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * gradient_accumulation_steps
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress bar update
        if (step + 1) % log_every_n_steps == 0:
            pbar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "acc": f"{correct/total*100:.2f}%"
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return {"train_loss": avg_loss, "train_accuracy": accuracy}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int = 5
) -> Dict[str, float]:
    """Evaluasi model pada validation/test set.
    
    Args:
        model: Model.
        data_loader: Val/test DataLoader.
        criterion: Loss function.
        device: Device.
        num_classes: Jumlah kelas.
        
    Returns:
        Dict metrik (loss, accuracy, F1, dll).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(pixel_values=images)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    
    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_f1_macro": f1_macro,
        "val_f1_weighted": f1_weighted,
        "val_precision_macro": precision_macro,
        "val_recall_macro": recall_macro,
    }


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str,
    experiment_name: str = "experiment",
    separate_backbone_lr: bool = False,
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """Full training loop dengan semua bells and whistles.
    
    Args:
        model: Model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training config.
        device: Device.
        experiment_name: Nama eksperimen.
        separate_backbone_lr: Discriminative LR backbone/head.
        class_weights: Optional class weights untuk weighted loss.
        
    Returns:
        Dict berisi best metrics dan training history.
    """
    train_config = config["training"]
    num_epochs = train_config["epochs"]
    
    # Loss function
    loss_config = config.get("loss", {})
    label_smoothing = loss_config.get("label_smoothing", 0.0)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Optimizer
    optimizer = create_optimizer(model, train_config, separate_backbone_lr)
    
    # Scheduler
    scheduler = create_scheduler(optimizer, train_config, num_epochs)
    
    # Mixed precision scaler
    scaler = GradScaler() if train_config.get("mixed_precision", False) and device == "cuda" else None
    
    # Early stopping
    es_config = train_config.get("early_stopping", {})
    early_stopping = None
    if es_config.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=es_config.get("patience", 10),
            mode=es_config.get("mode", "max")
        )
    
    # Checkpoint manager
    ckpt_config = config.get("checkpointing", {})
    checkpoint_mgr = CheckpointManager(
        save_dir=ckpt_config.get("save_dir", f"experiments/checkpoints/{experiment_name}"),
        monitor=ckpt_config.get("monitor", "val_f1_macro"),
        mode=es_config.get("mode", "max")
    )
    
    # CSV Logger
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", f"experiments/logs/{experiment_name}")
    csv_logger = CSVLogger(str(Path(log_dir) / "training_log.csv"))
    
    # TensorBoard (optional)
    tb_writer = None
    if log_config.get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("[trainer] TensorBoard not available, using CSV only")
    
    # Training history
    history = defaultdict(list)
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"Epochs: {num_epochs}, Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scaler=scaler,
            max_grad_norm=train_config.get("max_grad_norm", 1.0),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
            log_every_n_steps=log_config.get("log_every_n_steps", 10)
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics["val_f1_macro"])
        else:
            scheduler.step()
        
        # Combine metrics
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        all_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": current_lr,
            "epoch_time_s": round(epoch_time, 2)
        }
        
        # Log
        csv_logger.log(all_metrics)
        if tb_writer:
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(key, value, epoch)
        
        # Update history
        for key, value in all_metrics.items():
            history[key].append(value)
        
        # Print epoch summary
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Loss: {train_metrics['train_loss']:.4f} | "
              f"Val F1: {val_metrics['val_f1_macro']:.4f} | "
              f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Checkpoint
        checkpoint_mgr.save(
            model, optimizer, epoch, all_metrics,
            extra={"scheduler_state_dict": scheduler.state_dict()}
        )
        
        # Early stopping
        if early_stopping:
            monitor_value = val_metrics.get(es_config.get("monitor", "val_f1_macro"), 0)
            if early_stopping(monitor_value):
                print(f"\n[trainer] Early stopping at epoch {epoch}")
                break
    
    total_time = time.time() - start_time
    
    if tb_writer:
        tb_writer.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Training Complete: {experiment_name}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Best {ckpt_config.get('monitor', 'val_f1_macro')}: {checkpoint_mgr.best_value:.4f}")
    print(f"{'='*60}\n")
    
    return {
        "best_value": checkpoint_mgr.best_value,
        "total_time_s": total_time,
        "history": dict(history),
        "num_epochs_trained": len(history["epoch"]),
    }
