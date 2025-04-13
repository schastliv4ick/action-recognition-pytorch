import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

import dataloader
from dataloader import PeopleDataset

from torchvision import transforms
from torch.amp import GradScaler, autocast
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Улучшенная модель
class EnhancedYOLOLike(nn.Module):
    def __init__(self, num_classes=20):
        super(EnhancedYOLOLike, self).__init__()
        
        def conv_block(in_c, out_c, pool=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.1),
                SEBlock(out_c)  # SE блок
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
                layers.append(nn.Dropout2d(0.25))  # dropout
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256, pool=False)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Аугментации
train_transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((288, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_epoch(model, loader, optimizer, criterion, scaler, device):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for data, target in tqdm(loader, desc="Training"):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:  # mixed precision только для CUDA
                with autocast(device_type=device.type, dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Обычное обучение для CPU
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
        precision, recall, f1 = calculate_metrics(all_preds, all_targets)
        
        return avg_loss, accuracy, precision, recall, f1


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            
            with autocast(device_type=device.type, dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    precision, recall, f1 = calculate_metrics(all_preds, all_targets)
    
    return avg_loss, accuracy, precision, recall, f1


def calculate_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    return precision, recall, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        import os
        torch.set_num_threads(os.cpu_count())
        print(f"Using {os.cpu_count()} CPU threads")
    
    # Подготовка данных
    data_dir = "D:\\VS\\ml\\human_poses_data\\yandex-ml-2025\\data\\train_answers.csv"
    full_dataset = PeopleDataset(data_dir, transform=None)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform
    
    batch_size = 64  # Увеличенный batch size
    num_workers = 4
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    model = EnhancedYOLOLike(num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    scaler = None
    if device.type == 'cuda':
        scaler = GradScaler(device_type='cuda')
    else:
        print("CUDA not available, training on CPU without GradScaler")
    
    num_epochs = 30
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()