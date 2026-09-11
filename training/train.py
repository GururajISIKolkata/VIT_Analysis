import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            while isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=5,
    lr=1e-3,
    return_losses=False
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            while isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={acc:.2f}")

    if return_losses:
        return acc, train_losses

    return acc