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

exp_config = config["exp_5"]

epochs = exp_config["epochs"]
batch_size = exp_config["batch_size"]
learning_rate = float(exp_config["learning_rate"])
base_vit_config = exp_config["base_vit_config"]

print(f"Config:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")

train_loader, test_loader = get_dataloaders(
    data_dir="../data",
    batch_size=batch_size,
    fraction=1.0
)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# %%
# --- Cell 4: Experiment Loop ---
results = {
    "config": {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "base_vit_config": base_vit_config
    },
    "metrics": {}
}

# Train ViT (using existing training loop)
model = ViT(**base_vit_config)

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

# Collect attention maps from a test batch
model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    logits, attn_maps = model(images)

# Visualize: use first sample, average attention across heads for each layer
attention_maps_for_plot = []
for attn in attn_maps:
    layer_map = attn[0].mean(dim=0).detach().cpu().numpy()  # [tokens, tokens]
    attention_maps_for_plot.append(layer_map)

# Entropy computation per head and per layer: H = -sum(p*log(p))
entropy_per_layer = []
entropy_per_head = {}

for layer_idx, attn in enumerate(attn_maps):
    probs = attn.clamp(min=1e-12)
    head_entropy = -(probs * torch.log(probs)).sum(dim=-1)  # [B, heads, tokens]
    head_entropy_mean = head_entropy.mean(dim=(0, 2))  # [heads]

    layer_key = f"layer_{layer_idx + 1}"
    entropy_per_head[layer_key] = [float(x.item()) for x in head_entropy_mean]
    entropy_per_layer.append(float(head_entropy_mean.mean().item()))

results["metrics"] = {
    "test_accuracy": float(test_acc),
    "training_losses": [float(x) for x in train_losses],
    "training_time_seconds": float(training_time),
    "entropy_per_layer": entropy_per_layer,
    "entropy_per_head": entropy_per_head
}

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Training Time: {training_time:.2f}s")

# %%
# --- Cell 5: Save Results ---
save_results(results, "exp5_attention_entropy")
print("Saved JSON: ../Results/exp5_attention_entropy.json")

# %%
# --- Cell 6: Plot ---
# Plot 1: Attention maps for different layers
num_layers = len(attention_maps_for_plot)
fig1, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))

if num_layers == 1:
    axes = [axes]

for i in range(num_layers):
    im = axes[i].imshow(attention_maps_for_plot[i], cmap="viridis")
    axes[i].set_title(f"Layer {i+1}")
    axes[i].set_xlabel("Key Token Index")
    axes[i].set_ylabel("Query Token Index")
    fig1.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

fig1.suptitle("Experiment 5: Attention Maps Across Layers", y=1.03)
plt.tight_layout()
save_plot("exp5_attention_maps_layers", fig=fig1)
plt.show()

# Plot 2: Entropy vs layer depth
layer_depth = np.arange(1, len(results["metrics"]["entropy_per_layer"]) + 1)
layer_entropy = np.array(results["metrics"]["entropy_per_layer"])

fig2 = plt.figure(figsize=(8, 5))
plt.plot(layer_depth, layer_entropy, marker="o", linewidth=2)
plt.xlabel("Layer Depth")
plt.ylabel("Attention Entropy")
plt.title("Experiment 5: Entropy vs Layer Depth")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp5_entropy_vs_layer_depth", fig=fig2)
plt.show()

# Plot 3: Entropy per attention head (layer x head heatmap)
head_matrix = np.array([
    results["metrics"]["entropy_per_head"][f"layer_{i+1}"]
    for i in range(len(layer_depth))
])

fig3 = plt.figure(figsize=(8, 5))
plt.imshow(head_matrix, aspect="auto", cmap="magma")
plt.colorbar(label="Entropy")
plt.xlabel("Head Index")
plt.ylabel("Layer Index")
plt.title("Experiment 5: Entropy per Attention Head")
plt.xticks(ticks=np.arange(head_matrix.shape[1]), labels=np.arange(1, head_matrix.shape[1] + 1))
plt.yticks(ticks=np.arange(head_matrix.shape[0]), labels=np.arange(1, head_matrix.shape[0] + 1))
plt.grid(False)
plt.tight_layout()
save_plot("exp5_entropy_per_head_heatmap", fig=fig3)
plt.show()

print("Saved plot: ../Results/exp5_attention_maps_layers.png")
print("Saved plot: ../Results/exp5_entropy_vs_layer_depth.png")
print("Saved plot: ../Results/exp5_entropy_per_head_heatmap.png")


