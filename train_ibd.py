import os
import torch
import torch.nn as nn
from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader
from model import IBDClassifierCNN

def make_loaders(root="data", img_size=224, bs=32, num_workers=0):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs*2, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, train_ds.classes

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

def main(epochs=10, lr=3e-4, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_loaders(num_workers=0)

    model = IBDClassifierCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "ibd_cnn_best.pt")

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")
    print(f"Best val_acc: {best_val:.3f} (saved to ibd_cnn_best.pt)")

if __name__ == "__main__":
    torch.manual_seed(0)
    main(epochs=10) 
