from sklearn.model_selection import KFold
from tqdm import tqdm
import torch
from torch import device, cuda
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torchsummary import summary
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from utils.engine import setup_trainer, setup_evaluators
from utils.logging import setup_event_handlers, setup_metrics_history
from utils.plotting import plot_metrics, visualize_predictions

import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, Subset

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                   
            nn.Conv2d(channels, channels // reduction, 1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale
    

class PoseCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(PoseCNN, self).__init__()
        
        # Depthwise Separable Convolution блок
        def dws_conv(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, 
                          padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Основная архитектура с учетом большого размера изображения
        self.features = nn.Sequential(
            # Первый блок - обычная conv для сохранения информации
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 144x256
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Последующие блоки - depthwise separable
            dws_conv(16, 32, stride=2),  # 72x128
            dws_conv(32, 64, stride=2),  # 36x64
            dws_conv(64, 128, stride=2), # 18x32
            dws_conv(128, 256, stride=2), # 9x16
            
            # Дополнительные слои без downsampling
            dws_conv(256, 256),
            dws_conv(256, 256)
        )
        
        # Spatial Pyramid Pooling для учета разных аспектов позы
        self.spp = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256*4*4, 256),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.spp(x)
        x = self.classifier(x)
        return x


# Функция для ранней остановки
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class RandomAdjustColor:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        if random.random() < 0.5:
            img = F.adjust_brightness(img, random.uniform(1 - self.brightness, 1 + self.brightness))
        if random.random() < 0.5:
            img = F.adjust_contrast(img, random.uniform(1 - self.contrast, 1 + self.contrast))
        if random.random() < 0.5:
            img = F.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation))
        if random.random() < 0.5:
            img = F.adjust_hue(img, random.uniform(-self.hue, self.hue))
        return img
    
"""# Определение аугментации данных
transform = transforms.Compose([
    # Случайное горизонтальное отражение
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Случайное вращение изображения (до 30 градусов)
    transforms.RandomRotation(degrees=15),
    
    # Случайное изменение размера и обрезка
    transforms.RandomResizedCrop(size=(288, 512), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    
    # Случайное изменение цветовых характеристик
    RandomAdjustColor(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # Преобразование в тензор PyTorch
    transforms.ToTensor(),
    
    # Нормализация изображения (стандартные значения для ImageNet)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])"""

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),  # Увеличьте с 15
    transforms.RandomResizedCrop(size=(288, 512), scale=(0.6, 1.0)),  # Увеличьте разброс
    RandomAdjustColor(brightness=0.3, contrast=0.3),  # Увеличьте параметры
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PeopleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(os.path.join(data_dir, 'train_answers.csv'))
        self.img_dir = os.path.join(data_dir, 'img_train')
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        img_id = os.path.splitext(img_name)[0]
        label = self.labels_df[self.labels_df['img_id'].astype(str) == img_id]['target_feature'].values[0]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def setup_data_loaders(batch_size, train_set, valid_set=None, num_workers=4):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if valid_set is not None else None

    return train_loader, valid_loader


def print_batch_shape(data_loader: DataLoader, loader_type: str, ):
    valid_batch = next(iter(data_loader))
    images_valid, labels_valid = valid_batch
    print(f"{loader_type} batch shape: {images_valid.shape}")


def calculate_metrics(preds, targets):
    """Calculate precision, recall, f1 score"""
    preds = np.array(preds)
    targets = np.array(targets)

    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)

    return precision, recall, f1


def train_and_validate(model, train_loader, valid_loader, device, num_epochs=50):
    """Train and validate model for one fold"""
    num_classes = 20
    
    # Calculate class weights for current fold
    train_indices = train_loader.dataset.indices
    train_targets = [full_dataset[idx][1] for idx in train_indices]
    class_counts = [train_targets.count(i) for i in range(num_classes)]
    class_counts = [c if c != 0 else 1 for c in class_counts]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0.001)
    
    train_metrics_history, valid_metrics_history = setup_metrics_history()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_targets = []
        
        train_iterator = tqdm(train_loader, desc="Training", unit="batch")
        for batch_idx, (data, target) in enumerate(train_iterator):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(target.cpu().numpy())
            
            train_iterator.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        train_accuracy = 100. * np.sum(np.array(all_train_preds) == np.array(all_train_targets)) / len(all_train_targets)
        train_precision, train_recall, train_f1 = calculate_metrics(all_train_preds, all_train_targets)
        
        train_metrics_history['loss'].append(train_loss)
        train_metrics_history['accuracy'].append(train_accuracy)
        train_metrics_history['precision'].append(train_precision)
        train_metrics_history['recall'].append(train_recall)
        train_metrics_history['f1'].append(train_f1)
        
        # Validation phase
        if valid_loader:
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_targets = []
            
            valid_iterator = tqdm(valid_loader, desc="Validation", unit="batch")
            with torch.no_grad():
                for data, target in valid_iterator:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    
                    preds = torch.argmax(output, dim=1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_targets.extend(target.cpu().numpy())
                    
                    valid_iterator.set_postfix(val_loss=val_loss / len(valid_loader))
            
            val_loss /= len(valid_loader)
            val_accuracy = 100. * np.sum(np.array(all_val_preds) == np.array(all_val_targets)) / len(all_val_targets)
            val_precision, val_recall, val_f1 = calculate_metrics(all_val_preds, all_val_targets)
            
            scheduler.step(val_loss)
            print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}, Val Loss: {val_loss}")
            early_stopping(val_loss)
            
            valid_metrics_history['loss'].append(val_loss)
            valid_metrics_history['accuracy'].append(val_accuracy)
            valid_metrics_history['precision'].append(val_precision)
            valid_metrics_history['recall'].append(val_recall)
            valid_metrics_history['f1'].append(val_f1)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}%, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        if valid_loader:
            print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, "
                  f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    
    return train_metrics_history, valid_metrics_history


if __name__ == "__main__":
    device = device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Initialize the model
    model = PoseCNN(num_classes=20)
    model.to(device)
    summary(model, (3, 288, 512))
    print("\n")

    # Load the full dataset
    PATH_TO_DATA = "V:\ML\yandex-ml-2025\data"
    print("Loading the dataset...")
    full_dataset = PeopleDataset(PATH_TO_DATA)
    
    # Setup K-Fold cross validation
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    best_model = None
    best_val_f1 = 0.0
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{num_folds}")
        print(f"{'='*40}\n")
        
        # Create subsets for current fold
        train_subsampler = Subset(full_dataset, train_ids)
        val_subsampler = Subset(full_dataset, val_ids)
        
        # Apply transforms
        train_subsampler.dataset.transform = get_train_transforms()
        val_subsampler.dataset.transform = get_val_transforms()
        
        # Create data loaders
        BATCH_SIZE = 32
        train_loader, val_loader = setup_data_loaders(
            batch_size=BATCH_SIZE,
            train_set=train_subsampler,
            valid_set=val_subsampler
        )
        
        # Initialize new model for each fold
        model = PoseCNN(num_classes=20)
        model.to(device)
        
        # Train and validate
        train_metrics, val_metrics = train_and_validate(
            model=model,
            train_loader=train_loader,
            valid_loader=val_loader,
            device=device,
            num_epochs=50
        )
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model': model
        }
        fold_results.append(fold_result)
        
        # Track best model based on validation F1 score
        current_val_f1 = max(val_metrics['f1'])
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_model = model.state_dict()
            print(f"New best model found at fold {fold + 1} with val F1: {best_val_f1:.4f}")
    
    # After all folds, save the best model
    if best_model is not None:
        torch.save(best_model, 'best_model_crossval.pth')
        print("\nSaved best model based on validation F1 score")
    
    # Calculate and print average metrics across all folds
    avg_train_metrics = {
        'loss': np.mean([max(res['train_metrics']['loss']) for res in fold_results]),
        'accuracy': np.mean([max(res['train_metrics']['accuracy']) for res in fold_results]),
        'precision': np.mean([max(res['train_metrics']['precision']) for res in fold_results]),
        'recall': np.mean([max(res['train_metrics']['recall']) for res in fold_results]),
        'f1': np.mean([max(res['train_metrics']['f1']) for res in fold_results])
    }
    
    avg_val_metrics = {
        'loss': np.mean([max(res['val_metrics']['loss']) for res in fold_results]),
        'accuracy': np.mean([max(res['val_metrics']['accuracy']) for res in fold_results]),
        'precision': np.mean([max(res['val_metrics']['precision']) for res in fold_results]),
        'recall': np.mean([max(res['val_metrics']['recall']) for res in fold_results]),
        'f1': np.mean([max(res['val_metrics']['f1']) for res in fold_results])
    }
    
    print("\nCross-validation results summary:")
    print(f"Average Train Metrics - Loss: {avg_train_metrics['loss']:.4f}, "
          f"Acc: {avg_train_metrics['accuracy']:.2f}%, "
          f"Precision: {avg_train_metrics['precision']:.4f}, "
          f"Recall: {avg_train_metrics['recall']:.4f}, "
          f"F1: {avg_train_metrics['f1']:.4f}")
    
    print(f"Average Valid Metrics - Loss: {avg_val_metrics['loss']:.4f}, "
          f"Acc: {avg_val_metrics['accuracy']:.2f}%, "
          f"Precision: {avg_val_metrics['precision']:.4f}, "
          f"Recall: {avg_val_metrics['recall']:.4f}, "
          f"F1: {avg_val_metrics['f1']:.4f}")
    
    # Plot metrics for the last fold as an example
    print("\nPlotting metrics for the last fold...")
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    plot_metrics(fold_results[-1]['train_metrics'], fold_results[-1]['val_metrics'], metrics_to_plot=metrics_to_plot)
    
    # Visualize predictions from the last fold
    class_names = ['sports', 'inactivity quiet/light', 'miscellaneous', 'occupation', 'water activities',
                   'home activities', 'lawn and garden', 'religious activities', 'winter activities',
                   'conditioning exercise', 'bicycling', 'fishing and hunting', 'dancing', 'walking', 'running',
                   'self care', 'home repair', 'volunteer activities', 'music playing', 'transportation']
    
    print("\nVisualizing predictions from the last fold...")
    visualize_predictions(fold_results[-1]['model'], val_loader, device, class_names)