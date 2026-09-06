"""
download.py — Script untuk download dataset Cassava Leaf Disease.

Usage:
    python -m src.data.download --config configs/data.yaml

Jika dataset sudah ada di path yang dikonfigurasi, script akan skip download.
Jika kredensial Kaggle tidak tersedia, script berhenti dan minta path manual.
"""

import argparse
import os
import sys
from pathlib import Path

# Tambah project root ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config, resolve_path


def check_existing_data(raw_dir: Path) -> bool:
    """Cek apakah dataset sudah ada di folder raw.
    
    Args:
        raw_dir: Path ke folder data/raw atau dataset folder.
        
    Returns:
        True jika dataset sudah tersedia.
    """
    if not raw_dir.exists():
        return False
    
    # Cek apakah ada subfolder kelas penyakit
    expected_classes = ["cbb", "cbsd", "cgm", "cmd", "healthy"]
    found_classes = [
        d.name for d in raw_dir.iterdir() 
        if d.is_dir() and d.name in expected_classes
    ]
    
    if len(found_classes) == len(expected_classes):
        print(f"[download.py] Dataset ditemukan di {raw_dir}")
        for cls in expected_classes:
            count = len(list((raw_dir / cls).glob("*")))
            print(f"  - {cls}: {count} files")
        return True
    
    return False


def download_from_kaggle(target_dir: Path) -> None:
    """Download dataset dari Kaggle API.
    
    Args:
        target_dir: Path tujuan download.
    """
    # Cek apakah kaggle CLI tersedia
    try:
        import kaggle
    except ImportError:
        print("[download.py] ERROR: 'kaggle' package not installed.")
        print("  Install: pip install kaggle")
        print("  Atau upload dataset manual ke folder data/raw/")
        sys.exit(1)
    
    # Cek kredensial
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() and "KAGGLE_USERNAME" not in os.environ:
        print("[download.py] ERROR: Kaggle credentials not found.")
        print("  Opsi 1: Buat ~/.kaggle/kaggle.json")
        print("  Opsi 2: Set KAGGLE_USERNAME dan KAGGLE_KEY environment variables")
        print("  Opsi 3: Upload dataset manual ke folder data/raw/")
        print("\n  Dataset yang diperlukan:")
        print("  https://www.kaggle.com/competitions/cassava-leaf-disease-classification")
        sys.exit(1)
    
    # Download
    print(f"[download.py] Downloading dataset to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        "cassava-leaf-disease-classification",
        path=str(target_dir),
        quiet=False
    )
    
    print("[download.py] Download complete. Extracting...")
    import zipfile
    for zip_file in target_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(target_dir)
        zip_file.unlink()  # Hapus zip setelah extract
    
    print("[download.py] Extraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Download Cassava Leaf Disease dataset"
    )
    parser.add_argument(
        "--config", type=str, default="configs/data.yaml",
        help="Path to data config YAML"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    raw_dir = resolve_path(config["dataset"]["raw_dir"])
    
    print(f"[download.py] Checking dataset at: {raw_dir}")
    
    # Cek apakah data sudah ada
    if check_existing_data(raw_dir):
        print("[download.py] Dataset sudah tersedia. Skip download.")
        return
    
    # Coba download dari Kaggle
    print("[download.py] Dataset belum tersedia. Mencoba download...")
    download_from_kaggle(raw_dir)
    
    # Verifikasi setelah download
    if not check_existing_data(raw_dir):
        print("[download.py] WARNING: Dataset tidak terdeteksi setelah download.")
        print("  Pastikan struktur folder sesuai: raw_dir/{cbb, cbsd, cgm, cmd, healthy}/")
        sys.exit(1)
    
    print("[download.py] Done.")


if __name__ == "__main__":
    main()
