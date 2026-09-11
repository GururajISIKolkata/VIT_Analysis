from pathlib import Path
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from models.vit import ViT
from utils.dataset import get_dataloaders
from training.train import train_model
from utils.io import save_results, save_plot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_config = config["exp_3"]

epochs = exp_config["epochs"]
batch_size = exp_config["batch_size"]
learning_rate = float(exp_config["learning_rate"])
variants = exp_config["variants"]
base_vit_config = exp_config["base_vit_config"]

print(f"Config:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Variants: {[v['name'] for v in variants]}")
train_loader, test_loader = get_dataloaders(
    data_dir="../data",
    batch_size=batch_size,
    fraction=1.0
)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")


results = {
    "config": {
        "base_vit_config": base_vit_config,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "variants": variants
    },
    "variants": {}
}

for variant in variants:
    variant_name = variant["name"]
    use_cls_token = variant["use_cls_token"]

    print(f"\n==============================")
    print(f"Training: {variant_name}")
    print(f"==============================")

    # Create model
    vit_config = base_vit_config.copy()
    vit_config["use_cls_token"] = use_cls_token

    model = ViT(**vit_config)

    # Train and get loss curves
    start_time = time.time()
    test_acc, train_losses = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        return_losses=True,
        lr=learning_rate
    )
    training_time = time.time() - start_time

    # Store results
    results["variants"][variant_name] = {
        "use_cls_token": use_cls_token,
        "test_accuracy": test_acc,
        "training_losses": train_losses,
        "training_time_seconds": training_time
    }

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Training Time: {training_time:.2f}s")


fig, ax = plt.subplots(figsize=(8, 5))

variant_names = list(results["variants"].keys())
test_accs = [results["variants"][v]["test_accuracy"] for v in variant_names]

bars = ax.bar(variant_names, test_accs, color=['#1f77b4', '#ff7f0e'], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, test_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Experiment 3: CLS Token vs Mean Pooling - Test Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

save_plot("exp3_cls_vs_mean_accuracy", fig=fig)
plt.show()

print(f"Saved: exp3_cls_vs_mean_accuracy.png")


fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e']
for idx, (variant_name, variant_data) in enumerate(results["variants"].items()):
    losses = variant_data["training_losses"]
    epochs_range = range(1, len(losses) + 1)
    ax.plot(epochs_range, losses, marker='o', label=variant_name, linewidth=2, color=colors[idx], markersize=5)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Experiment 3: CLS Token vs Mean Pooling - Training Loss Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

save_plot("exp3_cls_vs_mean_loss", fig=fig)
plt.show()

print(f"Saved: exp3_cls_vs_mean_loss.png")


print("\n" + "="*50)
print("Experiment 3 Summary")
print("="*50)

for variant_name, variant_data in results["variants"].items():
    print(f"\n{variant_name}:")
    print(f"  Test Accuracy: {variant_data['test_accuracy']:.2f}%")
    print(f"  Training Time: {variant_data['training_time_seconds']:.2f}s")
    print(f"  Final Loss: {variant_data['training_losses'][-1]:.4f}")

accs = [results["variants"][v]["test_accuracy"] for v in results["variants"]]
diff = abs(accs[0] - accs[1])
print(f"\nAccuracy Difference: {diff:.2f}%")


save_results(results, "exp3_cls_vs_mean")

print("\nResults saved:")
print("  - Results/exp3_cls_vs_mean.json")
print("  - Results/exp3_cls_vs_mean_accuracy.png")
print("  - Results/exp3_cls_vs_mean_loss.png")


