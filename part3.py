import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
import pandas as pd
import math
from tqdm.auto import tqdm
import os

# =====================
# 配置部分 (Part3 Improved for 60% Target)
# =====================
CONFIG = {
    "project": "ds542-part3-transfer-improved",
    "seed": 42,
    "batch_size": 128,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",
    
    # 迁移学习相关配置
    "pretrained": True,           
    "freeze_epochs": 3,           # 缩短冻结阶段
    "unfreeze_epochs": 70,        # 延长解冻阶段，总训练轮次 73
    "base_lr": 0.01,
    "min_lr": 1e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.4,
    
    # OOD 测试数据目录（challenge 数据）
    "ood_dir": "./data/ood-test"
}

# =====================
# 模型定义 (使用 ResNet50 并适配 CIFAR-100)
# =====================
def get_model():
    """加载预训练 ResNet50 并适配 CIFAR-100，初始冻结除 fc 外的所有层"""
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    # 修改首层卷积，适应 CIFAR-100 32×32 输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 移除第一个最大池化，避免过度下采样
    model.maxpool = nn.Identity()
    # 替换最后的全连接层
    model.fc = nn.Linear(model.fc.in_features, 100)
    # 冻结除 fc 外的所有层
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return model.to(CONFIG['device'])

# =====================
# 数据增强
# =====================
def get_transforms():
    """设计针对迁移学习的数据增强策略"""
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test

# =====================
# MixUp 数据增强
# =====================
def mixup_data(x, y, alpha=0.4):
    """对输入进行 MixUp 增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# =====================
# 训练工具函数
# =====================
def train_epoch(model, loader, optimizer, criterion, epoch, phase):
    """训练或验证一个 epoch"""
    if phase == 'train':
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(loader, desc=f"{phase} Epoch {epoch}", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(CONFIG['device'], non_blocking=True)
        labels = labels.to(CONFIG['device'], non_blocking=True)
        
        # 仅在训练阶段使用 MixUp
        if phase == 'train' and CONFIG['mixup_alpha'] > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG['mixup_alpha'])
        
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            if phase == 'train' and CONFIG['mixup_alpha'] > 0:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
        
        if phase == 'train':
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += batch_size
        
        progress_bar.set_postfix({
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples
        })
    return total_loss / total_samples, total_correct / total_samples

# =====================
# 采用 OneCycleLR 调度器（用于 Phase 2）
# =====================
def get_onecycle_scheduler(optimizer, num_steps, max_lr):
    """返回 OneCycleLR 调度器"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        total_steps=num_steps, 
        pct_start=0.3, 
        anneal_strategy='cos', 
        div_factor=25.0, 
        final_div_factor=1e4
    )

# =====================
# 生成提交文件（调用 eval_ood 模块）
# =====================
def generate_submission(model, CONFIG):
    """
    使用 eval_ood 模块对 OOD 测试数据进行预测，并生成提交文件，
    格式与 sample_submission.csv 保持一致。
    """
    import eval_ood  # 确保 eval_ood.py 与本代码在同一目录
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv('submission_part3.csv', index=False)
    print("Submission file saved as submission_part3.csv")

# =====================
# 主训练流程
# =====================
def main():
    # 初始化随机种子和 WandB
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    wandb.init(project=CONFIG['project'], config=CONFIG)
    
    # 数据加载
    transform_train, transform_test = get_transforms()
    trainset = torchvision.datasets.CIFAR100(
        root=CONFIG['data_dir'], train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG['data_dir'], train=False, download=True, transform=transform_test)
    
    # 分割验证集（80/20）
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    
    # DataLoader（开启 pin_memory 与 persistent_workers 优化 IO）
    trainloader = DataLoader(
        trainset, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    valloader = DataLoader(
        valset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True)
    testloader = DataLoader(
        testset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # 初始化模型和损失函数
    model = get_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    # ===== Phase 1: 仅训练最后一层（冻结状态） =====
    print("\n=== Phase 1: Training Last Layer Only ===")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    # 使用 CosineAnnealingLR 调度器，周期为 freeze_epochs
    scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['freeze_epochs'])
    for epoch in range(CONFIG['freeze_epochs']):
        current_epoch = epoch + 1
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, current_epoch, 'train')
        val_loss, val_acc = train_epoch(model, valloader, optimizer, criterion, current_epoch, 'val')
        scheduler_obj.step()
        wandb.log({
            "epoch": current_epoch,
            "phase": "freeze",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        print(f"Epoch {current_epoch}/{CONFIG['freeze_epochs']} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # ===== Phase 2: 分阶段逐步解冻 =====
    print("\n=== Phase 2: Gradual Unfreezing ===")
    total_phase2_epochs = CONFIG['unfreeze_epochs']
    # 初始：解冻 layer4（保留 fc 已解冻），构造参数组：fc 与 layer4
    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
    param_groups = [
        {"params": model.fc.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 1.0},
        {"params": model.layer4.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.5},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
    # 使用 OneCycleLR 调度器：总步数 = num_steps_per_epoch * total_phase2_epochs
    num_steps = len(trainloader) * total_phase2_epochs
    scheduler_oc = get_onecycle_scheduler(optimizer, num_steps, max_lr=CONFIG['base_lr'])
    
    for epoch in range(total_phase2_epochs):
        current_epoch = CONFIG['freeze_epochs'] + epoch + 1
        
        # 当达到解冻阶段 1/2 时，解冻 layer3
        if epoch == total_phase2_epochs // 2:
            print(">> Unfreezing layer3")
            for name, param in model.named_parameters():
                if 'layer3' in name:
                    param.requires_grad = True
            param_groups = [
                {"params": model.fc.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 1.0},
                {"params": model.layer4.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.5},
                {"params": model.layer3.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.2},
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
            num_steps = len(trainloader) * (total_phase2_epochs - epoch)
            scheduler_oc = get_onecycle_scheduler(optimizer, num_steps, max_lr=CONFIG['base_lr'])
        
        # 当达到解冻阶段 3/4 时，解冻 layer2
        if epoch == (3 * total_phase2_epochs) // 4:
            print(">> Unfreezing layer2")
            for name, param in model.named_parameters():
                if 'layer2' in name:
                    param.requires_grad = True
            param_groups = [
                {"params": model.fc.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 1.0},
                {"params": model.layer4.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.5},
                {"params": model.layer3.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.2},
                {"params": model.layer2.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.1},
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
            num_steps = len(trainloader) * (total_phase2_epochs - epoch)
            scheduler_oc = get_onecycle_scheduler(optimizer, num_steps, max_lr=CONFIG['base_lr'])
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, current_epoch, 'train')
        val_loss, val_acc = train_epoch(model, valloader, optimizer, criterion, current_epoch, 'val')
        scheduler_oc.step()
        wandb.log({
            "epoch": current_epoch,
            "phase": "unfreeze",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        print(f"Epoch {current_epoch}/{CONFIG['freeze_epochs']+total_phase2_epochs} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型（依据验证准确率）
        if val_acc > wandb.run.summary.get("best_val_acc", 0):
            wandb.run.summary["best_val_acc"] = val_acc
            torch.save(model.state_dict(), "best_model_part3.pth")
    
    # ===== 最终评估与生成提交文件 =====
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load("best_model_part3.pth"))
    test_loss, test_acc = train_epoch(model, testloader, optimizer, criterion, 0, 'val')
    print(f"Test Accuracy: {test_acc:.2%}")
    wandb.log({"test_acc": test_acc})
    
    # 生成提交文件（调用 eval_ood 模块，生成符合要求的 CSV 文件）
    generate_submission(model, CONFIG)

if __name__ == '__main__':
    main()
