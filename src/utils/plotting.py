import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List
from torch import Tensor, no_grad, argmax
from torch import max as torch_max
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import config
from random import sample as random_sample
import math
from collections import defaultdict
import numpy as np
import os


def plot_metric(metric_name, subplot_num, epochs, train_metric_value, valid_metric_value, nrows, ncols, is_ylim=False):
    plt.subplot(nrows, ncols, subplot_num)

    # Check if the train_metric_value or valid_metric_value are empty before accessing
    if not train_metric_value:
        print(f"Warning: Missing data for train {metric_name}. Skipping plot for this metric.")
        return
    if not valid_metric_value:
        print(f"Warning: Missing data for valid {metric_name}. Skipping plot for this metric.")
        return

    # Convert tensors to numpy arrays if needed
    def convert_to_numpy(values):
        # If values are tensors, reduce them to a scalar (e.g., mean) or convert to numpy array
        if isinstance(values[0], Tensor):
            # Convert tensor to numpy array (taking mean if needed)
            values = [val.mean().item() if isinstance(val, Tensor) else val for val in values]
        return values

    train_metric_value = convert_to_numpy(train_metric_value)
    valid_metric_value = convert_to_numpy(valid_metric_value)

    # Check if the lengths of epochs and metric values match
    if len(train_metric_value) != len(epochs) or len(valid_metric_value) != len(epochs):
        print(f"Error: The number of epochs ({len(epochs)}) does not match the number of values for {metric_name}. "
              f"Skipping plot for this metric.")
        return

    # Plot the metric
    plt.plot(epochs, train_metric_value, label=f"Train {metric_name}", color='blue')
    plt.plot(epochs, valid_metric_value, label=f"Valid {metric_name}", color='orange')
    plt.title(metric_name)
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)

    # If y-limits are needed, set them
    if is_ylim:
        plt.ylim(0, 1)

    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def plot_metrics(train_metrics_history: defaultdict, valid_metrics_history: defaultdict,
                 metrics_to_plot: List[str], save_path=config.SAVE_DIR):
    if "loss" not in train_metrics_history or not train_metrics_history["loss"]:
        print("Error: No training loss data found!")
        return

    epochs = range(1, len(train_metrics_history["loss"]) + 1)
    metrics_to_plot_without_loss = [metric for metric in metrics_to_plot if metric != "loss"]
    total_metrics = len(metrics_to_plot_without_loss) + 1

    ncols = math.ceil(total_metrics ** 0.5)
    nrows = math.ceil(total_metrics / ncols)

    plt.figure(figsize=(4 * ncols, 4 * nrows))

    train_loss = train_metrics_history.get("loss", [])
    valid_loss = valid_metrics_history.get("loss", [])
    plot_metric("Loss", 1, epochs, train_loss, valid_loss, nrows, ncols)

    for idx, metric_name in enumerate(metrics_to_plot_without_loss, start=2):
        train_metric_value = train_metrics_history.get(metric_name, [])
        valid_metric_value = valid_metrics_history.get(metric_name, [])
        plot_metric(metric_name, idx, epochs, train_metric_value, valid_metric_value, nrows, ncols, is_ylim=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"metrics.png"), dpi=300, bbox_inches='tight')
        print("saved metrics")
    plt.close()


def plot_metric_and_loss(train_metrics_history: defaultdict, valid_metrics_history: defaultdict, metric_to_plot: str):
    # Ensure the 'loss' exists and is not empty for defining epochs
    if "loss" not in train_metrics_history or not train_metrics_history["loss"]:
        print("Error: No training loss data found!")
        return

    epochs = range(1, len(train_metrics_history["loss"]) + 1)

    # Check if the given metric is valid
    if metric_to_plot not in train_metrics_history or metric_to_plot not in valid_metrics_history:
        print(f"Error: The metric '{metric_to_plot}' is not found in the provided data.")
        return

    # Set up the plot size
    plt.figure(figsize=(10, 5))

    # Plot Train loss and Valid loss
    train_loss = train_metrics_history.get("loss", [])
    valid_loss = valid_metrics_history.get("loss", [])

    plot_metric("Loss", 1, epochs, train_loss, valid_loss, nrows=1, ncols=2)

    # Plot the specified additional metric (e.g., accuracy, precision, etc.)
    train_metric_value = train_metrics_history.get(metric_to_plot, [])
    valid_metric_value = valid_metrics_history.get(metric_to_plot, [])
    plot_metric(metric_to_plot, 2, epochs, train_metric_value, valid_metric_value, nrows=1, ncols=2, is_ylim=True)

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, valid_loader, device, class_names, num_images=15):
    model.eval()
    images, labels = next(iter(valid_loader))
    images, labels = images.to(device), labels.to(device)

    with no_grad():
        outputs = model(images)
        _, predicted = torch_max(outputs, 1)

    random_indices = random_sample(range(len(images)), num_images)
    fig, axes = plt.subplots(3, 5, figsize=(8, 5))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(images[idx].cpu().numpy().transpose(1, 2, 0))

        title = f"Pred: {class_names[predicted[idx].item()]}\nTrue: {class_names[labels[idx].item()]}"
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_first_images(dataset):
    plt.figure(figsize=(8, 6))
    for i in range(12):
        image, label = dataset[i]
        # Если использовались torchvision трансформации, то нужно преобразовать обратно в PIL или numpy
        if isinstance(image, Tensor):
            image = image.permute(1, 2, 0).numpy()  # CHW -> HWC
        plt.subplot(3, 4, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label.item() if isinstance(label, Tensor) else label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_metrics_per_class(model, data_loader, device, class_names, save_path: str = None):
    model.eval()
    all_preds = []
    all_targets = []

    with no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    if len(all_preds) == 0 or len(all_targets) == 0:
        print("Error: Empty predictions or targets.")
        return

    num_classes = len(class_names)
    class_accuracies = []
    class_precisions = []
    class_f1s = []

    for class_idx in range(num_classes):
        true_labels = np.array(all_targets)
        pred_labels = np.array(all_preds)
        class_mask = true_labels == class_idx

        if np.sum(class_mask) == 0:
            acc = prec = f1 = 0
        else:
            relevant_preds = pred_labels[class_mask]
            relevant_trues = true_labels[class_mask]
            acc = accuracy_score(relevant_trues, relevant_preds)
            prec = precision_score(true_labels, pred_labels, labels=[class_idx], average='macro', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, labels=[class_idx], average='macro', zero_division=0)

        class_accuracies.append(acc)
        class_precisions.append(prec)
        class_f1s.append(f1)

    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(16, 8))
    plt.bar(x - width, class_accuracies, width, label='Accuracy')
    plt.bar(x, class_precisions, width, label='Precision')
    plt.bar(x + width, class_f1s, width, label='F1 Score')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Accuracy, Precision, and F1 Score')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"distribution_per_classes.png"),
                    dpi=300,
                    bbox_inches='tight')
        print("saved distribustion")
    plt.close()
