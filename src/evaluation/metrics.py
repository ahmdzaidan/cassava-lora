"""
metrics.py — Fungsi-fungsi metrik evaluasi untuk klasifikasi.

Mendukung:
- Accuracy, Precision, Recall, F1 (macro & weighted)
- Confusion matrix
- Parameter count & efficiency metrics
"""

import numpy as np
from typing import Dict, List, Any, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    average_methods: List[str] = None
) -> Dict[str, Any]:
    """Hitung semua metrik klasifikasi.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Nama kelas.
        average_methods: List metode averaging (default: ["macro", "weighted"]).
        
    Returns:
        Dict berisi semua metrik.
    """
    if average_methods is None:
        average_methods = ["macro", "weighted"]
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }
    
    for avg in average_methods:
        metrics[f"precision_{avg}"] = precision_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"f1_{avg}"] = f1_score(
            y_true, y_pred, average=avg, zero_division=0
        )
    
    # Per-class metrics
    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )
    
    if class_names:
        for i, cls_name in enumerate(class_names):
            if i < len(per_class_f1):
                metrics[f"precision_{cls_name}"] = per_class_precision[i]
                metrics[f"recall_{cls_name}"] = per_class_recall[i]
                metrics[f"f1_{cls_name}"] = per_class_f1[i]
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: str = None
) -> Tuple[np.ndarray, List[str]]:
    """Hitung confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Nama kelas.
        normalize: 'true', 'pred', 'all', or None.
        
    Returns:
        Tuple (confusion_matrix, class_names).
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    if class_names is None:
        unique_labels = sorted(set(y_true) | set(y_pred))
        class_names = [str(l) for l in unique_labels]
    
    return cm, class_names


def get_classification_report_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> str:
    """Dapatkan classification report sebagai string.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Nama kelas.
        
    Returns:
        String classification report.
    """
    return classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        zero_division=0
    )


def count_parameters(model) -> Dict[str, int]:
    """Hitung jumlah parameter model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dict berisi total, trainable, frozen parameter count.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": round(trainable / total * 100, 4) if total > 0 else 0,
    }


def compute_efficiency_metrics(
    model_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Hitung metrik efisiensi: parameter vs performa.
    
    Args:
        model_results: List of dicts, masing-masing berisi:
            - name, trainable_params, total_params, accuracy, f1_macro
            
    Returns:
        List of dicts dengan tambahan efficiency metrics.
    """
    enriched = []
    for result in model_results:
        r = result.copy()
        
        trainable = r.get("trainable_params", 0)
        total = r.get("total_params", 1)
        accuracy = r.get("accuracy", 0)
        f1 = r.get("f1_macro", 0)
        
        # Efficiency: performa per juta trainable parameter
        r["params_millions"] = round(trainable / 1e6, 2)
        if trainable > 0:
            r["accuracy_per_M_params"] = round(accuracy / (trainable / 1e6), 4)
            r["f1_per_M_params"] = round(f1 / (trainable / 1e6), 4)
        else:
            r["accuracy_per_M_params"] = 0
            r["f1_per_M_params"] = 0
        
        r["compression_ratio"] = round(total / trainable, 2) if trainable > 0 else float("inf")
        
        enriched.append(r)
    
    return enriched
