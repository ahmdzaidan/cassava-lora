# Cassava LoRA MAE-ViT — Research Pipeline

**Analyzing the Effect of Low-Rank Adaptation (LoRA) on MAE-ViT Embedding Space in Cassava Leaf Disease Classification**

---

## 📋 Overview

An end-to-end research pipeline designed to investigate and analyze the structural and representation dynamics of Low-Rank Adaptation (LoRA) applied to Masked Autoencoder Vision Transformers (MAE-ViT) for Cassava Leaf Disease Classification. 

The pipeline evaluates three main research objectives:
- **T1: Classification Performance & Parameter Efficiency:** Evaluates accuracy, macro F1-score, per-class F1, trainable parameter count, memory usage, and execution speed across model variants.
- **T2: Embedding & Attention Matrix Structure Analysis:** Inspects weight updates ($\Delta W$), Singular Value Decomposition (SVD), effective rank, intruder dimensions, attention weight distributions, and representation similarity (Linear CKA).
- **T3: [CLS] Embedding Vector Shift & Clustering Quality:** Measures cosine distance shift of [CLS] tokens relative to initial/pre-trained embeddings, cluster separation metrics (Silhouette Score, Davies-Bouldin Index), 2D manifold projections (t-SNE, UMAP), and statistical correlation between rank $r$ and representation metrics.

### Dataset
- **Cassava Leaf Disease Classification** (Kaggle / Makerere AI Lab, ICLR 2020 Workshop)
- **5 Classes:** Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD), and Healthy.
- ~5,656 labeled cassava leaf images.

### Model Variants
- **Backbone:** ViT-Base/16 initialized with MAE pre-trained weights (`facebook/vit-mae-base`).
- **3 Evaluated Variants:** 
  1. Linear Probe (Frozen backbone, trainable classifier head)
  2. Full Fine-Tuning (All parameters trainable)
  3. LoRA Fine-Tuning (Rank grid: $r \in \{4, 8, 16, 32\}$ applied to Attention query/value projection layers)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to repository directory
cd cassava-lora-mae-vit

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, peft; print('Installation verified!')"
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

### 2. Dataset Preparation

Ensure the dataset is located at `../Dataset/train/train/` relative to the project root directory, structured as follows:

```
Dataset/train/train/
├── cbb/        # Cassava Bacterial Blight
├── cbsd/       # Cassava Brown Streak Disease
├── cgm/        # Cassava Green Mottle
├── cmd/        # Cassava Mosaic Disease
└── healthy/    # Healthy
```

*Note: If your dataset is stored elsewhere, update `dataset.raw_dir` in `configs/data.yaml`.*

### 3. Running the Pipeline (Phase by Phase)

```bash
# Phase 1: Data Preprocessing & CSV Splits
python -m src.data.preprocessing --config configs/data.yaml

# Phase 2: Linear Probe Baseline
python -m src.training.train_linear_probe --config configs/train_linear_probe.yaml

# Phase 3: Full Fine-Tuning Baseline
python -m src.training.train_full_finetune --config configs/train_full_finetune.yaml

# Phase 4: LoRA Fine-Tuning Grid Search (r=4, 8, 16, 32)
python -m src.training.train_lora --config configs/train_lora.yaml --rank 8
python -m src.training.train_lora --config configs/train_lora.yaml --rank 4
python -m src.training.train_lora --config configs/train_lora.yaml --rank 16
python -m src.training.train_lora --config configs/train_lora.yaml --rank 32

# Phase 5: Objective T1 Comparative Evaluation
python -m src.evaluation.evaluate --config configs/analysis.yaml --objective T1

# Phase 6: Objective T2 Structural Analysis (SVD & Attention)
python -m src.analysis.svd_weight_analysis --config configs/analysis.yaml
python -m src.analysis.attention_matrix_analysis --config configs/analysis.yaml

# Phase 7: Objective T3 Representation Shift & Clustering Analysis
python -m src.analysis.cls_embedding_shift --config configs/analysis.yaml
python -m src.analysis.clustering_metrics --config configs/analysis.yaml
python -m src.analysis.dimensionality_reduction --config configs/analysis.yaml

# Phase 9: Unit Tests
pytest tests/ -v
```

---

## 📁 Repository Structure

```
cassava-lora-mae-vit/
├── README.md                           # Project documentation (English)
├── plan.md                             # Comprehensive technical specification
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── configs/                            # YAML Configuration files
│   ├── data.yaml                       # Data loading and split configuration
│   ├── model_mae_vit.yaml              # MAE-ViT backbone setup
│   ├── train_full_finetune.yaml        # Full fine-tuning hyperparameters
│   ├── train_lora.yaml                 # LoRA hyperparameters & rank grid
│   ├── train_linear_probe.yaml         # Linear probing hyperparameters
│   └── analysis.yaml                   # Analysis & metric computation parameters
│
├── data/                               # Data storage (git-ignored)
│   ├── splits/                         # Stratified CSV splits (train/val/test)
│   └── processed/                      # Preprocessed caching directory
│
├── src/                                # Source code directory
│   ├── data/                           # Data processing modules
│   │   ├── download.py                 # Dataset download script
│   │   ├── preprocessing.py            # Train/val/test split generation
│   │   └── dataset.py                  # PyTorch Dataset & augmentations
│   │
│   ├── models/                         # Model architecture modules
│   │   ├── mae_vit.py                  # MAE-ViT backbone wrapper
│   │   ├── lora_layers.py              # LoRA implementation (PEFT & custom)
│   │   └── classifier_head.py          # Classification head module
│   │
│   ├── training/                       # Training execution modules
│   │   ├── trainer_utils.py            # Training loop, early stopping, logging
│   │   ├── train_linear_probe.py       # Phase 2 pipeline execution
│   │   ├── train_full_finetune.py      # Phase 3 pipeline execution
│   │   └── train_lora.py               # Phase 4 pipeline execution
│   │
│   ├── evaluation/                     # Metric evaluation modules
│   │   ├── metrics.py                  # Classification metric calculations
│   │   └── evaluate.py                 # T1 comparative evaluation runner
│   │
│   ├── analysis/                       # Analysis & probing modules
│   │   ├── svd_weight_analysis.py      # SVD & rank distribution on ΔW (T2)
│   │   ├── attention_matrix_analysis.py# Attention entropy & rollout maps (T2)
│   │   ├── cka.py                      # Centered Kernel Alignment (T2)
│   │   ├── cls_embedding_shift.py      # Cosine embedding shift analysis (T3)
│   │   ├── clustering_metrics.py       # Silhouette Score & DBI analysis (T3)
│   │   └── dimensionality_reduction.py # t-SNE & UMAP manifold visualization (T3)
│   │
│   ├── viz/                            # Visualization module
│   │   └── plotting.py                 # Standardized paper-ready figure generators
│   │
│   └── utils/                          # Helper utilities
│       ├── seed.py                     # Deterministic reproducibility seed setter
│       └── config.py                   # YAML configuration parser
│
├── notebooks/                          # Interactive Jupyter Notebooks
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis & Class Distribution
│   ├── 02_sanity_check_model.ipynb     # Model forwarding & shape verification
│   └── 03_result_exploration.ipynb     # Results inspection & interactive plots
│
├── experiments/                        # Experiment outputs (git-ignored)
│   ├── checkpoints/                    # Saved PyTorch model weights (.pt / PEFT)
│   └── logs/                           # TensorBoard & JSON execution logs
│
├── reports/                            # Publication & analysis artifacts
│   ├── figures/                        # Generated vector & raster plots (PNG + SVG)
│   ├── tables/                         # CSV and JSON summary tables
│   └── summary_report.md               # Markdown research findings summary
│
└── tests/                             # Unit testing suite
    ├── test_dataset.py                 # Data pipeline validation
    ├── test_lora_layers.py             # LoRA adaptation & shape validation
    └── test_analysis_functions.py      # Statistical and analytical function tests
```

---

## ⚙️ Configuration

The pipeline relies on structured YAML configurations located in `configs/`. Example for adjusting LoRA ranks:

```yaml
# configs/train_lora.yaml
lora:
  target_modules: ["query", "value"]
  rank_grid: [4, 8, 16, 32]
  default_rank: 8
  alpha_ratio: 2  # lora_alpha = alpha_ratio * rank
  dropout: 0.1
```

---

## 📊 Expected Outputs & Deliverables

All generated outputs are stored systematically in `reports/` and `data/`:

| Output Description | Target Location |
|-------------------|-----------------|
| Stratified Split Datasets | `data/splits/{train,val,test}.csv` |
| Dataset Class Distribution Table | `reports/tables/class_distribution.csv` |
| Model Weights & Checkpoints | `experiments/checkpoints/` |
| Training Execution Logs | `experiments/logs/` |
| T1 Performance & Efficiency Table | `reports/tables/comparison_T1.csv` |
| T2 SVD & Rank Dynamics Summary | `reports/tables/svd_summary.csv` |
| T3 Clustering & Representation Metrics | `reports/tables/clustering_metrics.csv` |
| High-Resolution Plots (PNG 300dpi + SVG) | `reports/figures/fig_T{1,2,3}_*.{png,svg}` |
| Final Synthesized Research Report | `reports/summary_report.md` |

---

## 🧪 Testing

Run automated unit tests to verify data pipelines, LoRA layer mechanics, and mathematical analysis modules:

```bash
# Execute full unit test suite
pytest tests/ -v

# Run specific module tests
pytest tests/test_lora_layers.py -v

# Run tests with code coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📚 References

- **MAE:** He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR.
- **LoRA:** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
- **CKA:** Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). *Similarity of Neural Network Representations Revisited*. ICML.
- **Dataset:** Mwebaze, E., et al. (2019). *Cassava Leaf Disease Classification*. Kaggle / Makerere AI Lab.

---

## 📝 License & Citation

This project is created for undergraduate thesis research (*Skripsi*). Feel free to use and adapt the code for academic and research purposes.

