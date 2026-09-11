import sys
sys.path.append("..")
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from models.vit import ViT
from utils.dataset import get_dataloaders
from training.train import train_model
from utils.io import save_results, save_plot


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_config = config["exp_2"]

patch_sizes = exp_config["patch_sizes"]
batch_size = exp_config["batch_size"]
epochs = exp_config["epochs"]
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

test_accuracies = []
training_times_sec = []

for patch_size in patch_sizes:
    print("\n==============================")
    print(f"Training ViT with patch size {patch_size}x{patch_size}")
    print("==============================")

    set_seed(42)

    vit_config = base_vit_config.copy()
    vit_config["patch_size"] = patch_size

    model = ViT(**vit_config)

    start_time = time.time()
    test_acc = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=learning_rate
    )
    elapsed = time.time() - start_time

    test_accuracies.append(test_acc)
    training_times_sec.append(elapsed)

    print(f"Patch {patch_size}x{patch_size} -> Test Accuracy: {test_acc:.2f}%")
    print(f"Patch {patch_size}x{patch_size} -> Training Time: {elapsed:.2f} sec")


# --- Cell 5: Save Results ---
results = {
    "config": {
        "experiment": "exp2_patch_size",
        "model": "ViT",
        "patch_sizes": patch_sizes,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "base_vit_config": base_vit_config,
        "device": str(device)
    },
    "metrics": {
        "test_accuracy": {
            str(p): float(a) for p, a in zip(patch_sizes, test_accuracies)
        },
        "training_time_sec": {
            str(p): float(t) for p, t in zip(patch_sizes, training_times_sec)
        }
    }
}

save_results(results, "exp2_patch_size")
print("Saved JSON: ../Results/exp2_patch_size.json")

# %%
# --- Cell 6: Plot ---
labels = [f"{p}x{p}" for p in patch_sizes]

fig = plt.figure(figsize=(7, 4))
plt.plot(labels, test_accuracies, marker="o", label="Test Accuracy")
plt.xlabel("Patch Size")
plt.ylabel("Test Accuracy (%)")
plt.title("Experiment 2: Accuracy vs Patch Size")
plt.legend()
plt.grid(True)
save_plot("exp2_patch_size_accuracy", fig)
plt.show()

fig = plt.figure(figsize=(7, 4))
plt.bar(labels, training_times_sec, label="Training Time")
plt.xlabel("Patch Size")
plt.ylabel("Training Time (sec)")
plt.title("Experiment 2: Training Time vs Patch Size")
plt.legend()
plt.grid(True, axis="y")
save_plot("exp2_patch_size_training_time", fig)
plt.show()

# %%
