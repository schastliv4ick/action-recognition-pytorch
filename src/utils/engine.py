import torch
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Precision, Recall, Accuracy, Fbeta, Loss
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
from tqdm import tqdm


def setup_trainer(model, optimizer, criterion, device):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    return trainer


def setup_evaluators(model, criterion, device):
    precision = Precision()
    recall = Recall()
    f1 = Fbeta(beta=1.0, average=False, precision=precision, recall=recall)

    metrics = {'accuracy': Accuracy(),
               'precision': precision,
               'recall': recall,
               'f1': f1,
               "loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    valid_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    return train_evaluator, valid_evaluator


def evaluate_model(model, test_loader, criterion, device, out_for_table=False):
    """Оценивает модель на тестовом наборе данных после обучения."""
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics

    if out_for_table:
        params_count = sum(p.numel() for p in model.parameters())
        print(f"| {params_count} | {metrics['accuracy']:.4f} | {metrics['loss']:.4f} |")
    else:
        print(f"Test Results: Accuracy = {metrics['accuracy']:.4f}, Loss = {metrics['loss']:.4f}")


def calculate_metrics(preds, targets):
    """Calculate precision, recall, f1 score"""
    preds = np.array(preds)
    targets = np.array(targets)

    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)

    return precision, recall, f1


def train_step(model, data, target, criterion, optimizer, device):
    model.train()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = torch.argmax(output, dim=1)
    return loss.item(), preds.cpu().numpy(), target.cpu().numpy()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    train_loss = 0
    all_train_preds = []
    all_train_targets = []
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", unit="batch")

    for batch_idx, (data, target) in enumerate(train_iterator):
        loss, preds, targets = train_step(model, data, target, criterion, optimizer, device)
        train_loss += loss
        all_train_preds.extend(preds)
        all_train_targets.extend(targets)
        train_iterator.set_postfix(loss=loss)

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * np.sum(np.array(all_train_preds) == np.array(all_train_targets)) / len(all_train_targets)
    precision, recall, f1 = calculate_metrics(all_train_preds, all_train_targets)

    return avg_loss, accuracy, precision, recall, f1


def calculate_epoch_metrics(model, valid_loader, criterion, device):
    """Validates one epoch of model"""
    model.eval()

    val_loss = 0
    all_val_preds = []
    all_val_targets = []
    valid_iterator = tqdm(valid_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for data, target in valid_iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            val_loss += loss
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_targets.extend(target.cpu().numpy())
            valid_iterator.set_postfix(val_loss=loss / len(valid_loader))

    avg_loss = val_loss / len(valid_loader)
    accuracy = 100. * np.sum(np.array(all_val_preds) == np.array(all_val_targets)) / len(all_val_targets)
    precision, recall, f1 = calculate_metrics(all_val_preds, all_val_targets)

    metrics_data = avg_loss, accuracy, precision, recall, f1
    return metrics_data
