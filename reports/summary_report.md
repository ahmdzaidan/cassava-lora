# Summary Report — Pipeline Penelitian LoRA MAE-ViT

> **Status:** Template — akan diperbarui setelah semua eksperimen selesai dijalankan.

---

## Tujuan 1 (T1): Pengaruh LoRA terhadap Performa Klasifikasi

### Ringkasan
Perbandingan performa tiga varian model pada dataset Cassava Leaf Disease Classification:

| Model | Trainable Params | % of Total | Accuracy | Macro F1 | Training Time |
|-------|:-------:|:--------:|:--------:|:--------:|:-------:|
| Linear Probe | — | — | — | — | — |
| Full Fine-tune | — | — | — | — | — |
| LoRA r=4 (Q,V) | — | — | — | — | — |
| LoRA r=8 (Q,V) | — | — | — | — | — |
| LoRA r=16 (Q,V) | — | — | — | — | — |
| LoRA r=32 (Q,V) | — | — | — | — | — |

> **Catatan:** Tabel akan diisi otomatis oleh `src/evaluation/evaluate.py` setelah training selesai.

### Figure References
- `reports/figures/fig_T1_accuracy_vs_params.{png,svg}` — Trade-off accuracy vs parameter trainable
- `reports/figures/fig_T1_f1_comparison.{png,svg}` — Perbandingan F1 score
- `reports/figures/fig_T1_confusion_matrices.{png,svg}` — Grid confusion matrix

---

## Tujuan 2 (T2): Perubahan Struktur Ruang Embedding/Attention akibat LoRA

### Ringkasan
Analisis SVD pada ΔW = B·A menunjukkan:
- **Peluruhan nilai singular:** _(akan diisi)_
- **Effective rank per layer:** _(akan diisi)_
- **Intruder dimensions:** _(akan diisi)_
- **Perubahan entropi attention:** _(akan diisi)_

### Figure References
- `reports/figures/fig_T2_svd_decay_per_layer.{png,svg}` — Kurva peluruhan singular values
- `reports/figures/fig_T2_intruder_dimensions.{png,svg}` — Deteksi intruder dimensions
- `reports/figures/fig_T2_attention_heatmap_comparison.{png,svg}` — Perbandingan attention map
- `reports/figures/fig_T2_attention_entropy.{png,svg}` — Entropi attention per layer

### Table References
- `reports/tables/svd_summary.csv` — Ringkasan SVD per layer

---

## Tujuan 3 (T3): Hubungan Pergeseran Vektor [CLS] dengan Performa

### Ringkasan
Analisis pergeseran [CLS] embedding:
- **Pola shift per layer:** _(delayed specialization — akan diverifikasi)_
- **Silhouette Score:** before LoRA = _, after LoRA = _
- **Davies-Bouldin Index:** before LoRA = _, after LoRA = _
- **Korelasi shift vs akurasi per kelas:** _(akan diisi)_

### Figure References
- `reports/figures/fig_T3_cls_shift_per_layer.{png,svg}` — Cosine shift per layer
- `reports/figures/fig_T3_tsne_before_after.{png,svg}` — t-SNE visualisasi
- `reports/figures/fig_T3_umap_before_after.{png,svg}` — UMAP visualisasi

### Table References
- `reports/tables/clustering_metrics.csv` — Silhouette & DBI sebelum/sesudah
- `reports/tables/correlation_shift_vs_accuracy.csv` — Korelasi statistik

---

## Reproduksi

Untuk mereproduksi hasil penelitian, ikuti urutan perintah di `README.md` atau lihat `plan.md` Section 7.
