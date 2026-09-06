"""
config.py — Utility untuk load dan merge YAML config files.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load konfigurasi dari file YAML.
    
    Args:
        config_path: Path ke file YAML.
        
    Returns:
        Dictionary berisi konfigurasi.
        
    Raises:
        FileNotFoundError: Jika file config tidak ditemukan.
        yaml.YAMLError: Jika file YAML tidak valid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge beberapa config dictionaries. Config terakhir menang jika ada konflik.
    
    Args:
        *configs: Variable number of config dictionaries.
        
    Returns:
        Merged dictionary.
    """
    merged = {}
    for config in configs:
        merged = _deep_merge(merged, config)
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dua dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_project_root() -> Path:
    """Dapatkan root directory project (tempat requirements.txt berada).
    
    Returns:
        Path ke project root.
    """
    current = Path(__file__).resolve()
    # Naik dari src/utils/ ke project root
    root = current.parent.parent.parent
    
    # Validasi: cek apakah requirements.txt ada di root
    if not (root / "requirements.txt").exists():
        # Fallback: cari ke atas sampai menemukan requirements.txt
        for parent in current.parents:
            if (parent / "requirements.txt").exists():
                return parent
        raise FileNotFoundError(
            "Cannot find project root (no requirements.txt found in parent dirs)"
        )
    
    return root


def resolve_path(path: str, relative_to: Optional[str] = None) -> Path:
    """Resolve path relatif terhadap project root atau path tertentu.
    
    Args:
        path: Path yang akan di-resolve.
        relative_to: Base path. Jika None, gunakan project root.
        
    Returns:
        Absolute Path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    
    if relative_to is not None:
        base = Path(relative_to)
    else:
        base = get_project_root()
    
    return (base / p).resolve()
