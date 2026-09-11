DS-265 DLCV 2026: Assignment 2
====================================

Author: Ritik Kumar Badiya

---

## Overview
This repository contains code and experiments for analyzing Vision Transformer (ViT) and CNN models on CIFAR-10, as per the assignment requirements. The code is organized for modular experimentation and reproducibility.

---

## Directory Structure
- `experiments/` : Scripts for each experiment (exp1_data_efficiency.py, exp2_patch_size.py, ...)
- `models/`      : Model definitions (cnn.py, vit.py, patch_embedding.py, transformer.py)
- `training/`    : Training utilities (train.py)
- `utils/`       : Dataset loading, I/O utilities (dataset.py, io.py)
- `experiment/config.yaml`  : Configuration file for experiments

---

## Requirements
- **Python version:** 3.8+
- **Required packages:**
  - torch (>=1.10)
  - torchvision (>=0.11)
  - numpy
  - matplotlib
  - tqdm
  - pyyaml
  - seaborn, scikit-learn for analysis/plots

Install requirements with:
```
pip install torch torchvision numpy matplotlib tqdm pyyaml
```

---

## How to Run
1. **Clone the repository and navigate to the root/experiment folder.**
2. **Run any experiment:**
   ```
   python exp1_data_efficiency.py
   # or
   python exp2_patch_size.py
   # ...and so on for other experiments
   ```
3. **Configuration:**
   - Edit `experiments/config.yaml` to change hyperparameters or experiment settings.

---


## Contact
For queries, contact: ritikbadiya@iisc.ac.in
