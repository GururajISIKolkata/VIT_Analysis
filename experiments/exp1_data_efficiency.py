from pathlib import Path
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from models.cnn import ResNet18
from models.vit import ViT
from utils.dataset import get_dataloaders
from utils.io import save_results, save_plot
from training.train import train_model
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_config = config["exp_1"]

epochs = exp_config["epochs"]
batch_size = exp_config["batch_size"]
learning_rate_vit = float(exp_config["learning_rate_vit"])
learning_rate_cnn = float(exp_config["learning_rate_cnn"])
fractions = exp_config["fractions"]
base_vit_config = exp_config["base_vit_config"]

cnn_results = []
vit_results = []

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed()

print(f"Config:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate (ViT): {learning_rate_vit}")
print(f"  Learning rate (CNN): {learning_rate_cnn}")

for frac in fractions:
    print(f"\n==============================")
    print(f"Training with {int(frac*100)}% data")
    print(f"==============================")

    train_loader, test_loader = get_dataloaders(
        data_dir="../data",
        batch_size=batch_size,
        fraction=frac
    )

    # -------- CNN --------
    print("\n--- CNN (ResNet18) ---")
    cnn_model = ResNet18()

    cnn_acc = train_model(
        cnn_model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,   # start with 2 if slow
        lr=learning_rate_cnn
    )

    cnn_results.append(cnn_acc)

    # -------- ViT --------
    print("\n--- ViT ---")
    vit_model = ViT(**base_vit_config)

    vit_acc = train_model(
        vit_model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,   # start with 2 if slow
        lr=learning_rate_vit
    )

    vit_results.append(vit_acc)


fractions_pct = [int(frac * 100) for frac in fractions]

fig = plt.figure(figsize=(6,4))

plt.plot(fractions_pct, cnn_results, marker='o', label="CNN (ResNet18)")
plt.plot(fractions_pct, vit_results, marker='s', label="ViT")

plt.xlabel("Training Data (%)")
plt.ylabel("Test Accuracy (%)")
plt.title("Experiment 1: Data Efficiency (CNN vs ViT)")

plt.legend()
plt.grid()

save_plot("exp1_data_efficiency_accuracy_plot", fig=fig)
plt.show()

exp1_results = {
    "fractions": fractions,
    "fractions_percent": [int(frac * 100) for frac in fractions],
    "cnn_test_accuracy": cnn_results,
    "vit_test_accuracy": vit_results
}

save_results(exp1_results, "exp1_data_efficiency_results")

print("Saved plot: Results/exp1_data_efficiency_accuracy_plot.png")
print("Saved results: Results/exp1_data_efficiency_results.json")


