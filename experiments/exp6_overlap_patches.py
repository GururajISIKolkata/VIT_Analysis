# %%
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/Othercomputers/My PC (1)/IISC/2 SEM/DLCV/Assignment 02/DLCV-assignment02

# %%
# --- Cell 2: Imports ---
from pathlib import Path
import sys
sys.path.append("..")

import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import yaml

from models.vit import ViT
from utils.dataset import get_dataloaders
from training.train import train_model
from utils.io import save_results, save_plot

# Device and reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Ensure outputs are stored in ../Results/
results_dir = Path("../Results")
results_dir.mkdir(parents=True, exist_ok=True)

# %%
# --- Cell 3: Config ---
# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_config = config["exp_6"]

epochs = exp_config["epochs"]
batch_size = exp_config["batch_size"]
learning_rate = float(exp_config["learning_rate"])

print(f"Config:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
base_vit_config = exp_config["base_vit_config"]
stride_variants = exp_config["stride_variants"]

train_loader, test_loader = get_dataloaders(
    data_dir="../data",
    batch_size=batch_size,
    fraction=1.0
)

print(f"Stride variants: {stride_variants}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# %%
# --- Cell 4: Experiment Loop ---
results = {
    "config": {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "base_vit_config": base_vit_config,
        "stride_variants": stride_variants
    },
    "metrics": {}
}


for variant_name, stride_value in stride_variants.items():
    print("\n" + "=" * 50)
    print(f"Training variant: {variant_name} | stride={stride_value}")
    print("=" * 50)

    variant_config = dict(base_vit_config)
    variant_config["stride"] = stride_value

    model = ViT(**variant_config)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    test_acc, train_losses = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=learning_rate,
        return_losses=True
    )
    training_time = time.time() - start_time

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = 0.0

    results["metrics"][variant_name] = {
        "stride": int(stride_value),
        "test_accuracy": float(test_acc),
        "training_time_seconds": float(training_time),
        "gpu_peak_memory_mb": float(peak_memory_mb),
        "training_losses": [float(x) for x in train_losses]
    }

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.2f}s")
    print(f"GPU Peak Memory (MB): {peak_memory_mb:.2f}")

# %%
# --- Cell 5: Save Results ---
save_results(results, "exp6_overlap_patches")
print("Saved JSON: ../Results/exp6_overlap_patches.json")

# %%
# --- Cell 6: Plot ---
labels = list(stride_variants.keys())
accuracies = [results["metrics"][k]["test_accuracy"] for k in labels]
train_times = [results["metrics"][k]["training_time_seconds"] for k in labels]
peak_memory = [results["metrics"][k]["gpu_peak_memory_mb"] for k in labels]

# Plot 1: Test Accuracy
fig1 = plt.figure(figsize=(7, 5))
plt.bar(labels, accuracies, color=["#1f77b4", "#ff7f0e"])
plt.xlabel("Patch Strategy")
plt.ylabel("Test Accuracy (%)")
plt.title("Experiment 6: Accuracy (Overlapping vs Non-overlapping)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp6_overlap_patches_accuracy", fig=fig1)
plt.show()

# Plot 2: Training Time
fig2 = plt.figure(figsize=(7, 5))
plt.bar(labels, train_times, color=["#2ca02c", "#d62728"])
plt.xlabel("Patch Strategy")
plt.ylabel("Training Time (s)")
plt.title("Experiment 6: Training Time (Overlapping vs Non-overlapping)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp6_overlap_patches_training_time", fig=fig2)
plt.show()

# Plot 3: GPU Peak Memory
fig3 = plt.figure(figsize=(7, 5))
plt.bar(labels, peak_memory, color=["#9467bd", "#8c564b"])
plt.xlabel("Patch Strategy")
plt.ylabel("GPU Peak Memory (MB)")
plt.title("Experiment 6: GPU Peak Memory (Overlapping vs Non-overlapping)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp6_overlap_patches_gpu_memory", fig=fig3)
plt.show()

print("Saved plot: ../Results/exp6_overlap_patches_accuracy.png")
print("Saved plot: ../Results/exp6_overlap_patches_training_time.png")
print("Saved plot: ../Results/exp6_overlap_patches_gpu_memory.png")


