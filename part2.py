import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import wandb
import numpy as np

# -------------------
# Config
# -------------------
CONFIG = {
    "project": "ds542-part2-cnn",
    "batch_size": 128,
    "epochs": 30,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "val_split": 0.2,
    "data_dir": "./data"
}

# -------------------
# Seed
# -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# -------------------
# Model
# -------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 64x8x8

            nn.Conv2d(64, 128, 3, padding=1), # 128x8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 128x4x4

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG["dropout_rate"]),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        return self.net(x)

# -------------------
# Transforms
# -------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# -------------------
# Dataset & DataLoader
# -------------------
full_trainset = torchvision.datasets.CIFAR100(
    root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)

val_size = int(CONFIG["val_split"] * len(full_trainset))
train_size = len(full_trainset) - val_size
trainset, valset = random_split(full_trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)

testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

# -------------------
# Training
# -------------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def eval_model(model, dataloader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

# -------------------
# Main
# -------------------
def main():
    wandb.init(project=CONFIG["project"], config=CONFIG)

    model = SimpleCNN().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion)
        val_loss, val_acc = eval_model(model, valloader, criterion)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_part2.pth")
            wandb.run.summary["best_val_acc"] = best_val_acc

    # Final test
    model.load_state_dict(torch.load("best_model_part2.pth"))
    test_loss, test_acc = eval_model(model, testloader, criterion)
    print(f"Test Accuracy: {test_acc:.2%}")
    wandb.log({"test_acc": test_acc})

    wandb.finish()

if __name__ == "__main__":
    main()
