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

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_config = config["exp_4"]

epochs = exp_config["epochs"]
batch_size = exp_config["batch_size"]
learning_rate = float(exp_config["learning_rate"])
base_vit_config = exp_config["base_vit_config"]

# Experiment variable: positional encoding type
positional_variants = exp_config["positional_variants"]

train_loader, test_loader = get_dataloaders(
    data_dir="../data",
    batch_size=batch_size,
    fraction=1.0
)

print(f"Config:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"Positional variants: {positional_variants}")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

results = {
    "config": {
        "epochs": epochs,
        "batch_size": batch_size,
        "base_vit_config": base_vit_config,
        "learning_rate": learning_rate,
        "positional_variants": positional_variants
    },
    "metrics": {}
}

for pos_type in positional_variants:
    print("\n" + "=" * 50)
    print(f"Training ViT with pos_embed_type = {pos_type}")
    print("=" * 50)

    variant_config = dict(base_vit_config)
    variant_config["pos_embed_type"] = pos_type

    model = ViT(**variant_config)

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

    results["metrics"][pos_type] = {
        "test_accuracy": float(test_acc),
        "training_losses": [float(x) for x in train_losses],
        "training_time_seconds": float(training_time)
    }

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.2f}s")


save_results(results, "exp4_positional_encoding")
print("Saved JSON: ../Results/exp4_positional_encoding.json")


labels = positional_variants
accuracies = [results["metrics"][k]["test_accuracy"] for k in labels]

fig1 = plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.xlabel("Positional Encoding Type")
plt.ylabel("Test Accuracy (%)")
plt.title("Experiment 4: Positional Encoding Ablation - Test Accuracy")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp4_positional_encoding_accuracy", fig=fig1)
plt.show()

# Plot 2: Training stability (loss curves)
fig2 = plt.figure(figsize=(8, 5))
for k in labels:
    losses = results["metrics"][k]["training_losses"]
    epochs_axis = list(range(1, len(losses) + 1))
    plt.plot(epochs_axis, losses, marker="o", label=k)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Experiment 4: Positional Encoding Ablation - Training Stability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp4_positional_encoding_loss_curves", fig=fig2)
plt.show()

print("Saved plot: ../Results/exp4_positional_encoding_accuracy.png")
print("Saved plot: ../Results/exp4_positional_encoding_loss_curves.png")


