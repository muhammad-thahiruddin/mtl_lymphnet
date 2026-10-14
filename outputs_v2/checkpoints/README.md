## Model Checkpoints (Available Upon Reasonable Request)

This directory is intentionally kept empty (structure only) in version control.

### Why Weights Are Not in Git
The trained checkpoint files (`.pt`) in this directory total roughly **2.4 GB across 88 files** — well beyond standard Git repository limits (GitHub flags repositories over ~1 GB and rejects individual files exceeding 100 MB).

### How to Request Checkpoints
All trained model checkpoints supporting the published manuscript are preserved and available for academic replication and non-commercial research **upon reasonable request**.

To request the checkpoint files:
1. Contact the authors (**Muhammad Thahiruddin** at `muhammad.thahiruddin@ua.ac.id` or **Imam Wahyudi** at `imamwahyudi@ua.ac.id`), or open a GitHub Issue with the label `model-request`.
2. Provide a brief explanation of the intended academic/research purpose and institutional affiliation.
3. Specify whether you need the full checkpoint archive or specific experiment weights (see convention below).

Once received, place the `.pt` files directly into this directory (`outputs_v2/checkpoints/`).

### Checkpoint File Naming Conventions

The checkpoint files match the experiment names used throughout `outputs_v2/experiment_results.pkl` and the manuscript:

| Checkpoint Prefix | Description | Count |
|---|---|---|
| `exp1_mtl_main_seed{42,43,44}_fold{0-4}.pt` | Experiment 1: Primary proposed MTL model (5-fold nested CV, 3 random seeds) | 15 files |
| `exp3_seg_only_fold{0-4}.pt` | Experiment 3: Single-task segmentation baseline (U-Net with TinyViT-5M) | 5 files |
| `exp3_cls_only_fold{0-4}.pt` | Experiment 3: Single-task classification baseline (TinyViT-5M) | 5 files |
| `exp4_MobileNetV3-Large_fold{0-4}.pt` | Experiment 4: Backbone ablation (MobileNetV3-Large encoder) | 5 files |
| `exp4_EfficientNet-B0_fold{0-4}.pt` | Experiment 4: Backbone ablation (EfficientNet-B0 encoder) | 5 files |
| `exp5_Fixed_0.5_0.5_fold{0-4}.pt` | Experiment 5: Loss weighting ablation (Fixed equal weights $\lambda_{seg}=0.5, \lambda_{cls}=0.5$) | 5 files |
| `exp5_Fixed_0.7_0.3_fold{0-4}.pt` | Experiment 5: Loss weighting ablation (Fixed unequal weights $\lambda_{seg}=0.7, \lambda_{cls}=0.3$) | 5 files |
| `exp5_DWA_fold{0-4}.pt` | Experiment 5: Loss weighting ablation (Dynamic Weight Average, $T=2$) | 5 files |
| `ext_C1_to_C2__*` / `ext_C2_to_C1__*` | Experiment 2: Bidirectional external validation checkpoints (all variants across 3 seeds) | 38 files |
