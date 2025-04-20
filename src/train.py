from tqdm import tqdm
import torch
from torch import device, cuda
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torchsummary import summary
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from models.__all_models import *
from models.PoseCNNv2 import PoseCNNv2
import dataloader
from dataloader import PeopleDataset

from utils.engine import setup_trainer, setup_evaluators
from utils.logging import setup_event_handlers, setup_metrics_history
from utils import plotting

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import config


def calculate_metrics(preds, targets):
    """Calculate precision, recall, f1 score"""
    preds = np.array(preds)
    targets = np.array(targets)

    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)

    return precision, recall, f1


if __name__ == "__main__":
    device = device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = PoseCNNsc(num_classes=20)
    model.to(device)
    summary(model, (3, 288, 512))
    print("\n")

    """Preparing the data"""
    train_transforms = dataloader.get_transforms(augmentation_type=config.TRAIN_AUGMENTATION_TYPE)
    valid_transforms = dataloader.get_transforms(augmentation_type=config.VALID_AUGMENTATION_TYPE)

    print("Loading the dataset...")
    full_dataset = PeopleDataset(config.PATH_TO_DATA)

    train_set, valid_set = dataloader.split_dataset(full_dataset, valid_ratio=0.2)
    train_set.dataset.transform = train_transforms
    valid_set.dataset.transform = valid_transforms

    # Showing first 12 images after transforming them
    # plotting.show_first_images(full_dataset)

    print(f"Setting up data loaders with batch_size={config.BATCH_SIZE}...")
    train_loader, valid_loader = dataloader.setup_data_loaders(
        batch_size=config.BATCH_SIZE,
        train_set=train_set,
        valid_set=valid_set
    )

    """Training setup"""
    num_classes = 20

    train_indices = train_set.indices
    train_targets = [full_dataset[idx][1] for idx in train_indices]
    class_counts = [train_targets.count(i) for i in range(num_classes)]
    class_counts = [c if c != 0 else 1 for c in class_counts]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config.NUM_EPOCHS,  # Полный цикл на всё обучение
    eta_min=1e-6              # Минимальный LR
    )

    criterion = CrossEntropyLoss(weight=class_weights.to(device))

    print("Initializing trainer and evaluators...")
    trainer = setup_trainer(model, optimizer, criterion, device)
    train_evaluator, valid_evaluator = setup_evaluators(model, criterion, device)

    train_metrics_history, valid_metrics_history = setup_metrics_history()

    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_iterator = tqdm(train_loader, desc="Training", unit="batch")
        model.train()

        train_loss = 0
        all_train_preds = []
        all_train_targets = []

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

        scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy = 100. * np.sum(np.array(all_train_preds) == np.array(all_train_targets)) / len(
            all_train_targets)
        train_precision, train_recall, train_f1 = calculate_metrics(all_train_preds, all_train_targets)

        train_metrics_history['loss'].append(train_loss)
        train_metrics_history['accuracy'].append(train_accuracy)
        train_metrics_history['precision'].append(train_precision)
        train_metrics_history['recall'].append(train_recall)
        train_metrics_history['f1'].append(train_f1)

        if valid_loader:
            valid_iterator = tqdm(valid_loader, desc="Validation", unit="batch")
            model.eval()

            val_loss = 0
            all_val_preds = []
            all_val_targets = []

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

            valid_metrics_history['loss'].append(val_loss)
            valid_metrics_history['accuracy'].append(val_accuracy)
            valid_metrics_history['precision'].append(val_precision)
            valid_metrics_history['recall'].append(val_recall)
            valid_metrics_history['f1'].append(val_f1)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}%, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        if valid_loader:
            print(f"Valid - Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%,"
                  f" Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    """Results visualization""" 
    print("\nTraining completed!")
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    plotting.plot_metrics(train_metrics_history, valid_metrics_history, metrics_to_plot=metrics_to_plot)

    # To plot loss and one metric
    # plot_metric_and_loss(train_metrics_history, valid_metrics_history, "accuracy")

    class_names = ['sports', 'inactivity quiet/light', 'miscellaneous', 'occupation', 'water activities',
                   'home activities', 'lawn and garden', 'religious activities', 'winter activities',
                   'conditioning exercise', 'bicycling', 'fishing and hunting', 'dancing', 'walking', 'running',
                   'self care', 'home repair', 'volunteer activities', 'music playing', 'transportation']
    # plotting.visualize_predictions(model, valid_loader, device, class_names)

    plotting.plot_metrics_per_class(model, valid_loader, device, class_names)

    # evaluate_model(model, test_loader, criterion, device)
