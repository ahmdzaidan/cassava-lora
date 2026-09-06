"""
cka.py — Centered Kernel Alignment (CKA) — opsional, cross-check Tujuan 2.

Mengukur kemiripan representasi layer antara backbone-only vs backbone+LoRA.
CKA tinggi = representasi mirip, CKA rendah = representasi berubah signifikan.

Referensi: Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Hitung linear kernel K = X @ X^T.
    
    Args:
        X: Matrix of shape (n_samples, n_features).
        
    Returns:
        Kernel matrix of shape (n_samples, n_samples).
    """
    return X @ X.T


def center_kernel(K: np.ndarray) -> np.ndarray:
    """Center kernel matrix (HSIC centering).
    
    Args:
        K: Kernel matrix of shape (n, n).
        
    Returns:
        Centered kernel matrix.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Hitung Linear CKA antara dua representasi.
    
    Args:
        X: Representasi pertama, shape (n_samples, d1).
        Y: Representasi kedua, shape (n_samples, d2).
        
    Returns:
        CKA similarity score (0 = tidak mirip, 1 = identik).
    """
    # Kernel matrices
    K = linear_kernel(X)
    L = linear_kernel(Y)
    
    # Center
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    
    # HSIC
    hsic_kl = np.trace(K_c @ L_c)
    hsic_kk = np.trace(K_c @ K_c)
    hsic_ll = np.trace(L_c @ L_c)
    
    # CKA
    cka = hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-10)
    
    return float(cka)


def compute_cka_matrix(
    representations_1: List[np.ndarray],
    representations_2: List[np.ndarray],
    layer_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Hitung CKA matrix antar semua layer dari dua model.
    
    CKA[i,j] = CKA(model1_layer_i, model2_layer_j)
    
    Args:
        representations_1: List representasi per layer dari model 1.
        representations_2: List representasi per layer dari model 2.
        layer_names: Nama layer (opsional).
        
    Returns:
        Tuple (cka_matrix, layer_names).
    """
    n_layers_1 = len(representations_1)
    n_layers_2 = len(representations_2)
    
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(max(n_layers_1, n_layers_2))]
    
    cka_matrix = np.zeros((n_layers_1, n_layers_2))
    
    for i in range(n_layers_1):
        for j in range(n_layers_2):
            X = representations_1[i]
            Y = representations_2[j]
            
            if isinstance(X, torch.Tensor):
                X = X.detach().cpu().numpy()
            if isinstance(Y, torch.Tensor):
                Y = Y.detach().cpu().numpy()
            
            # Flatten jika 3D+
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            if Y.ndim > 2:
                Y = Y.reshape(Y.shape[0], -1)
            
            cka_matrix[i, j] = linear_cka(X, Y)
    
    return cka_matrix, layer_names


def extract_layer_representations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    max_samples: int = 500
) -> List[np.ndarray]:
    """Ekstrak representasi dari setiap layer model.
    
    Args:
        model: ViT model.
        dataloader: DataLoader.
        device: Device.
        max_samples: Maksimum sampel.
        
    Returns:
        List of numpy arrays, satu per layer (termasuk embedding layer).
    """
    model.eval()
    all_hidden_states = None
    total_samples = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            if total_samples >= max_samples:
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
            
            hidden_states = outputs.hidden_states  # tuple of tensors
            
            if all_hidden_states is None:
                all_hidden_states = [[] for _ in range(len(hidden_states))]
            
            for i, hs in enumerate(hidden_states):
                # Ambil [CLS] token
                cls_hs = hs[:, 0, :].cpu().numpy()
                all_hidden_states[i].append(cls_hs)
            
            total_samples += images.shape[0]
    
    # Concatenate
    representations = [
        np.concatenate(layer_reps, axis=0)[:max_samples]
        for layer_reps in all_hidden_states
    ]
    
    print(f"[cka] Extracted representations from {len(representations)} layers, "
          f"{representations[0].shape[0]} samples")
    
    return representations
