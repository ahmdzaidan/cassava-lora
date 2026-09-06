# Cassava LoRA MAE-ViT — Pipeline Penelitian

**Analisis Pengaruh Low-Rank Adaptation (LoRA) terhadap Ruang Embedding MAE-ViT pada Klasifikasi Penyakit Daun Singkong**

---

## 📋 Ringkasan

Pipeline penelitian end-to-end untuk menganalisis pengaruh LoRA (Low-Rank Adaptation) terhadap model MAE-ViT (Masked Autoencoder Vision Transformer) dalam tugas klasifikasi penyakit daun singkong. Pipeline mencakup:

- **T1:** Evaluasi performa klasifikasi (accuracy, F1, parameter efficiency)
- **T2:** Analisis struktur matriks embedding/attention (SVD, intruder dimensions, attention maps)
- **T3:** Analisis pergeseran vektor [CLS] (cosine shift, Silhouette/DBI, t-SNE/UMAP, korelasi statistik)

### Dataset
- **Cassava Leaf Disease Classification** (Kaggle, ICLR 2020 Workshop)
- 5 kelas: CBB, CBSD, CGM, CMD, Healthy
- ~5,656 citra daun singkong berlabel

### Model
- **Backbone:** ViT-Base/16 dengan MAE pretraining (`facebook/vit-mae-base`)
- **3 Varian:** Linear Probe | Full Fine-tune | LoRA (rank 4/8/16/32)

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/navigate ke repository
cd cassava-lora-mae-vit

# Install dependencies
pip install -r requirements.txt

# Verifikasi instalasi
python -c "import torch, transformers, peft; print('OK')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### 2. Siapkan Dataset

Dataset harus berada di `../Dataset/train/train/` (relatif terhadap project root) dengan struktur:
```
Dataset/train/train/
├── cbb/        # Cassava Bacterial Blight
├── cbsd/       # Cassava Brown Streak Disease
├── cgm/        # Cassava Green Mottle
├── cmd/        # Cassava Mosaic Disease
└── healthy/    # Healthy
```

Jika dataset di lokasi berbeda, edit `configs/data.yaml` → `dataset.raw_dir`.

### 3. Jalankan Pipeline (per Fase)

```bash
# Fase 1: Preprocessing (buat split CSV)
python -m src.data.preprocessing --config configs/data.yaml

# Fase 2: Linear Probe
python -m src.training.train_linear_probe --config configs/train_linear_probe.yaml

# Fase 3: Full Fine-tuning
python -m src.training.train_full_finetune --config configs/train_full_finetune.yaml

# Fase 4: LoRA (jalankan per rank, atau semua sekaligus)
python -m src.training.train_lora --config configs/train_lora.yaml --rank 8
python -m src.training.train_lora --config configs/train_lora.yaml --rank 4
python -m src.training.train_lora --config configs/train_lora.yaml --rank 16
python -m src.training.train_lora --config configs/train_lora.yaml --rank 32

# Fase 5: Evaluasi T1
python -m src.evaluation.evaluate --config configs/analysis.yaml --objective T1

# Fase 6: Analisis T2
python -m src.analysis.svd_weight_analysis --config configs/analysis.yaml
python -m src.analysis.attention_matrix_analysis --config configs/analysis.yaml

# Fase 7: Analisis T3
python -m src.analysis.cls_embedding_shift --config configs/analysis.yaml
python -m src.analysis.clustering_metrics --config configs/analysis.yaml
python -m src.analysis.dimensionality_reduction --config configs/analysis.yaml

# Fase 9: Tests
pytest tests/ -v
```

---

## 📁 Struktur Repository

```
cassava-lora-mae-vit/
├── README.md                           # Dokumentasi ini
├── plan.md                             # Spesifikasi teknis penelitian
├── requirements.txt                    # Dependencies Python
├── .gitignore
│
├── configs/                            # Konfigurasi YAML
│   ├── data.yaml                       # Data pipeline
│   ├── model_mae_vit.yaml             # Backbone architecture
│   ├── train_full_finetune.yaml       # Full fine-tuning hyperparams
│   ├── train_lora.yaml                # LoRA hyperparams + rank grid
│   ├── train_linear_probe.yaml        # Linear probe hyperparams
│   └── analysis.yaml                  # Analysis parameters
│
├── data/                              # Data (not committed)
│   ├── splits/                        # train.csv, val.csv, test.csv
│   └── processed/
│
├── src/                               # Source code
│   ├── data/                          # Data pipeline
│   │   ├── download.py               # Dataset download
│   │   ├── preprocessing.py          # Split & distribution
│   │   └── dataset.py                # PyTorch Dataset + augmentation
│   │
│   ├── models/                        # Model definitions
│   │   ├── mae_vit.py                # MAE-ViT backbone
│   │   ├── lora_layers.py            # LoRA implementation (PEFT + custom)
│   │   └── classifier_head.py        # Classification head
│   │
│   ├── training/                      # Training scripts
│   │   ├── trainer_utils.py          # Training loop, early stopping
│   │   ├── train_linear_probe.py     # Fase 2
│   │   ├── train_full_finetune.py    # Fase 3
│   │   └── train_lora.py            # Fase 4
│   │
│   ├── evaluation/                    # Evaluation
│   │   ├── metrics.py                # Classification metrics
│   │   └── evaluate.py               # Comparative evaluation (T1)
│   │
│   ├── analysis/                      # Analysis modules
│   │   ├── svd_weight_analysis.py    # SVD on ΔW (T2)
│   │   ├── attention_matrix_analysis.py  # Attention maps (T2)
│   │   ├── cka.py                    # Linear CKA (T2, optional)
│   │   ├── cls_embedding_shift.py    # CLS cosine shift (T3)
│   │   ├── clustering_metrics.py     # Silhouette/DBI (T3)
│   │   └── dimensionality_reduction.py  # t-SNE/UMAP (T3)
│   │
│   ├── viz/                           # Visualization
│   │   └── plotting.py               # All plot functions
│   │
│   └── utils/                         # Utilities
│       ├── seed.py                    # Global seed setter
│       └── config.py                  # YAML config loader
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_sanity_check_model.ipynb   # Model sanity check
│   └── 03_result_exploration.ipynb   # Results summary
│
├── experiments/                       # Experiment outputs (not committed)
│   ├── checkpoints/                  # Model checkpoints
│   └── logs/                         # Training logs
│
├── reports/                           # Reports & outputs
│   ├── figures/                      # Generated figures (PNG + SVG)
│   ├── tables/                       # Generated CSV/JSON tables
│   └── summary_report.md            # Summary of findings
│
└── tests/                            # Unit tests
    ├── test_dataset.py               # Data pipeline tests
    ├── test_lora_layers.py           # LoRA layer tests
    └── test_analysis_functions.py    # Analysis function tests
```

---

## ⚙️ Konfigurasi

Semua hyperparameter dikonfigurasi via YAML di `configs/`. Contoh mengubah LoRA rank:

```yaml
# configs/train_lora.yaml
lora:
  rank_grid: [4, 8, 16, 32]
  default_rank: 8
  alpha_ratio: 2  # alpha = 2 * rank
```

---

## 📊 Output

Setelah pipeline selesai, output tersedia di:

| Output | Lokasi |
|--------|--------|
| Split CSV | `data/splits/{train,val,test}.csv` |
| Class distribution | `reports/tables/class_distribution.csv` |
| Training logs | `experiments/logs/` |
| Model checkpoints | `experiments/checkpoints/` |
| Comparison table (T1) | `reports/tables/comparison_T1.csv` |
| SVD summary (T2) | `reports/tables/svd_summary.csv` |
| Clustering metrics (T3) | `reports/tables/clustering_metrics.csv` |
| All figures | `reports/figures/fig_T{1,2,3}_*.{png,svg}` |
| Summary report | `reports/summary_report.md` |

---

## 🧪 Testing

```bash
# Run semua tests
pytest tests/ -v

# Run specific test file
pytest tests/test_lora_layers.py -v

# Run dengan coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📚 Referensi

- **MAE:** He et al. (2022). "Masked Autoencoders Are Scalable Vision Learners"
- **LoRA:** Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models"
- **CKA:** Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited"
- **Dataset:** Mwebaze et al. (2019). Cassava Leaf Disease Classification (Kaggle/Makerere AI Lab)

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan skripsi / penelitian akademik.
