from pathlib import Path
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

exp_config = config["exp_7"]

vit_epochs = exp_config["vit_epochs"]
probe_epochs = exp_config["probe_epochs"]
batch_size = exp_config["batch_size"]
learning_rate = float(exp_config["learning_rate"])

print(f"Config:")
print(f"  ViT Epochs: {vit_epochs}")
print(f"  Probe Epochs: {probe_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
base_vit_config = exp_config["base_vit_config"]

train_loader, test_loader = get_dataloaders(
    data_dir="../data",
    batch_size=batch_size,
    fraction=1.0
)

num_layers = base_vit_config["depth"]
layer_indices = list(range(num_layers))

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
print(f"Layer indices for probes: {[i + 1 for i in layer_indices]}")


results = {
    "config": {
        "vit_epochs": vit_epochs,
        "probe_epochs": probe_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "base_vit_config": base_vit_config
    },
    "metrics": {}
}

# Stage 1: Train ViT using existing training loop
vit_model = ViT(**base_vit_config)

vit_start = time.time()
vit_test_acc, vit_train_losses = train_model(
    vit_model,
    train_loader,
    test_loader,
    device,
    epochs=vit_epochs,
    lr=learning_rate,
    return_losses=True
)
vit_train_time = time.time() - vit_start

# Freeze ViT parameters for linear probing
vit_model = vit_model.to(device)
vit_model.eval()
for p in vit_model.parameters():
    p.requires_grad = False

# Helper: extract one layer representation for all samples in a loader
def extract_layer_features(model, loader, target_layer_idx, device):
    feature_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            x = model.patch_embed(images)
            bsz = x.size(0)

            if model.use_cls_token:
                cls_tokens = model.cls_token.expand(bsz, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

            if model.pos_embed is not None:
                x = x + model.pos_embed.to(x.device)

            for layer_idx, layer in enumerate(model.encoder.layers):
                x, _ = layer(x)
                if layer_idx == target_layer_idx:
                    if model.use_cls_token:
                        rep = x[:, 0]
                    else:
                        rep = x.mean(dim=1)
                    feature_list.append(rep.detach().cpu())
                    label_list.append(labels.detach().cpu())
                    break

    features = torch.cat(feature_list, dim=0)
    targets = torch.cat(label_list, dim=0)
    return features, targets

# Linear probe model
class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

layer_accuracies = []
layer_training_times = []

# Stage 2: Train linear probe on each layer representation
for layer_idx in layer_indices:
    print("\n" + "=" * 50)
    print(f"Linear probe on Layer {layer_idx + 1}")
    print("=" * 50)

    train_features, train_labels = extract_layer_features(vit_model, train_loader, layer_idx, device)
    test_features, test_labels = extract_layer_features(vit_model, test_loader, layer_idx, device)

    probe_train_ds = TensorDataset(train_features, train_labels)
    probe_test_ds = TensorDataset(test_features, test_labels)

    probe_train_loader = DataLoader(probe_train_ds, batch_size=batch_size, shuffle=True)
    probe_test_loader = DataLoader(probe_test_ds, batch_size=batch_size, shuffle=False)

    probe_model = LinearProbe(in_dim=train_features.shape[1])

    probe_start = time.time()
    probe_acc, probe_losses = train_model(
        probe_model,
        probe_train_loader,
        probe_test_loader,
        device,
        epochs=probe_epochs,
        lr=learning_rate,
        return_losses=True
    )
    probe_time = time.time() - probe_start

    layer_accuracies.append(float(probe_acc))
    layer_training_times.append(float(probe_time))

    results["metrics"][f"layer_{layer_idx + 1}"] = {
        "classification_accuracy": float(probe_acc),
        "probe_training_time_seconds": float(probe_time),
        "probe_training_losses": [float(x) for x in probe_losses]
    }

results["metrics"]["summary"] = {
    "vit_test_accuracy": float(vit_test_acc),
    "vit_training_time_seconds": float(vit_train_time),
    "layer_indices": [int(i + 1) for i in layer_indices],
    "layer_accuracies": layer_accuracies,
    "layer_probe_training_time_seconds": layer_training_times
}

print(f"\nBase ViT Test Accuracy: {vit_test_acc:.2f}%")


save_results(results, "exp7_layerwise_features")
print("Saved JSON: ../Results/exp7_layerwise_features.json")


layer_ids = results["metrics"]["summary"]["layer_indices"]
accuracies = results["metrics"]["summary"]["layer_accuracies"]

fig = plt.figure(figsize=(8, 5))
plt.plot(layer_ids, accuracies, marker="o", linewidth=2)
plt.xlabel("Layer Index")
plt.ylabel("Classification Accuracy (%)")
plt.title("Experiment 7: Layer-wise Representation Quality")
plt.xticks(layer_ids)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
save_plot("exp7_layerwise_accuracy", fig=fig)
plt.show()

print("Saved plot: ../Results/exp7_layerwise_accuracy.png")


