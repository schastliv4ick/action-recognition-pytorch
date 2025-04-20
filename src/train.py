import torch
from torch import device, cuda
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss
from torchsummary import summary

from models.__all_models import *
import dataloader
from dataloader import PeopleDataset

from utils.engine import setup_trainer, setup_evaluators, train_epoch_and_get_metrics_dict, calculate_epoch_metrics
from utils.logging import setup_metrics_history, add_metrics_to_history, print_epoch_summary, save_best_models
from utils import plotting

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import config

if __name__ == "__main__":

    RESULTS_DIR = os.path.join("results", config.__name__)
    print("RESULTS_DIR", RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)


    device = device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = DensePoseCNN(num_classes=20)
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

    if config.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=25, eta_min=1e-6)
    elif config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config.NUM_EPOCHS, eta_min=0)

    criterion = CrossEntropyLoss(weight=class_weights.to(device))

    print("Initializing trainer and evaluators...")
    trainer = setup_trainer(model, optimizer, criterion, device)
    train_evaluator, valid_evaluator = setup_evaluators(model, criterion, device)

    train_metrics_history, valid_metrics_history = setup_metrics_history()

    best_valid_loss = float('inf')
    best_valid_f1 = 0.0
    model_name = model.__class__.__name__

    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_metrics_dict = train_epoch_and_get_metrics_dict(model, train_loader, criterion, optimizer, device, epoch, config.NUM_EPOCHS)
        scheduler.step()
        add_metrics_to_history(train_metrics_history, train_metrics_dict)

        valid_metrics_dict = {}
        if valid_loader:
            valid_metrics_dict = calculate_epoch_metrics(model, valid_loader, criterion, device)
            add_metrics_to_history(valid_metrics_history, valid_metrics_dict)
            best_valid_loss, best_valid_f1 = save_best_models(
                current_metrics=valid_metrics_dict,
                model=model,
                model_name=model_name,
                best_loss=best_valid_loss, 
                best_f1=best_valid_f1
            )

        print_epoch_summary(epoch, train_metrics_dict, valid_metrics_dict)

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

    plotting.plot_metrics_per_class(model, valid_loader, device, class_names, save_path=RESULTS_DIR)

    # evaluate_model(model, test_loader, criterion, device)
