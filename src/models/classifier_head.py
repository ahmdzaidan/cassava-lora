"""
classifier_head.py — Classification head di atas [CLS] token output ViT.

Mendukung:
- Linear head (single layer)
- MLP head (multi-layer dengan dropout)
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearClassifierHead(nn.Module):
    """Simple linear classifier head.
    
    Mengambil [CLS] token output dan memetakan ke num_classes.
    
    Args:
        hidden_size: Dimensi output backbone (ViT hidden_size, default 768).
        num_classes: Jumlah kelas (5 untuk Cassava).
        dropout: Dropout rate sebelum classifier.
    """
    
    def __init__(
        self, 
        hidden_size: int = 768, 
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [CLS] token embedding of shape (batch_size, hidden_size).
            
        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x = self.dropout(x)
        return self.classifier(x)


class MLPClassifierHead(nn.Module):
    """Multi-layer perceptron classifier head.
    
    Args:
        hidden_size: Dimensi input dari backbone.
        num_classes: Jumlah kelas.
        mlp_hidden_dim: Dimensi hidden layer MLP.
        num_layers: Jumlah hidden layers.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 5,
        mlp_hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = hidden_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ])
            in_dim = mlp_hidden_dim
        
        layers.append(nn.Linear(mlp_hidden_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [CLS] token embedding of shape (batch_size, hidden_size).
            
        Returns:
            Logits of shape (batch_size, num_classes).
        """
        return self.mlp(x)


class CassavaClassifier(nn.Module):
    """Full model: ViT backbone + classifier head.
    
    Menggabungkan backbone ViT dengan classification head, mengambil
    output [CLS] token dari backbone dan memprosesnya.
    
    Args:
        backbone: ViT backbone model (ViTModel dari HuggingFace).
        head: Classification head (LinearClassifierHead atau MLPClassifierHead).
    """
    
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> dict:
        """Forward pass melalui backbone + head.
        
        Args:
            pixel_values: Input images of shape (batch_size, 3, H, W).
            output_attentions: Jika True, return attention weights per layer.
            output_hidden_states: Jika True, return hidden states per layer.
            
        Returns:
            Dict berisi logits, dan opsional attentions/hidden_states.
        """
        # Forward melalui backbone
        backbone_output = self.backbone(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Ambil [CLS] token (index 0 dari sequence output)
        cls_embedding = backbone_output.last_hidden_state[:, 0, :]
        
        # Forward melalui head
        logits = self.head(cls_embedding)
        
        result = {"logits": logits, "cls_embedding": cls_embedding}
        
        if output_attentions:
            result["attentions"] = backbone_output.attentions
        if output_hidden_states:
            result["hidden_states"] = backbone_output.hidden_states
        
        return result
    
    def get_cls_embeddings_per_layer(
        self, 
        pixel_values: torch.Tensor
    ) -> list:
        """Dapatkan [CLS] embedding dari setiap layer (untuk analisis T3).
        
        Args:
            pixel_values: Input images.
            
        Returns:
            List of tensors, satu per layer, shape (batch_size, hidden_size).
        """
        backbone_output = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        # hidden_states: tuple of (embedding_output, layer_1, ..., layer_n)
        cls_per_layer = []
        for hidden_state in backbone_output.hidden_states:
            cls_per_layer.append(hidden_state[:, 0, :])  # [CLS] token
        
        return cls_per_layer


def build_classifier_head(config: dict) -> nn.Module:
    """Factory function untuk membuat classifier head dari config.
    
    Args:
        config: Model config dictionary (bagian 'classifier').
        
    Returns:
        Classification head module.
    """
    head_type = config.get("head_type", "linear")
    hidden_size = config.get("hidden_size", 768)
    num_classes = config.get("num_classes", 5)
    dropout = config.get("dropout", 0.1)
    
    if head_type == "linear":
        return LinearClassifierHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )
    elif head_type == "mlp":
        mlp_config = config.get("mlp", {})
        return MLPClassifierHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            mlp_hidden_dim=mlp_config.get("hidden_dim", 256),
            num_layers=mlp_config.get("num_layers", 2),
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
