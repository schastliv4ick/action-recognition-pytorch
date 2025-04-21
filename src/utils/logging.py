from torch import Tensor
from ignite.engine import Events
from ignite.handlers import ReduceLROnPlateauScheduler
import torch
import os
import config
from collections import defaultdict


def log_iteration_loss(engine):
    print(f"Epoch[{engine.state.epoch}] - Iter[{engine.state.iteration}]: loss = {engine.state.output}")


def run_evaluators_on_epoch(train_evaluator, valid_evaluator, train_loader, valid_loader):
    train_evaluator.run(train_loader)
    valid_evaluator.run(valid_loader)


def log_and_save_epoch_results(engine, label, metrics_history, silent=False):
    metrics = engine.state.metrics
    metrics_items = metrics.items()
    result = ', '.join([
        f"{m} = {v.mean().item():.4f}" if isinstance(v, Tensor) and v.numel() > 1 else f"{m} = {v.item():.4f}"
        if isinstance(v, Tensor) else f"{m} = {v:.4f}"
        for m, v in metrics_items
    ])
    if not silent:
        print(f"{label}: {result}")

    for metric_name, value in metrics_items:
        metrics_history[metric_name].append(value)


def setup_metrics_history():
    train_metrics_history = defaultdict(list)
    valid_metrics_history = defaultdict(list)
    return train_metrics_history, valid_metrics_history


def add_metrics_to_history(metrics_history, metrics_data: dict):
    metrics_history['loss'].append(metrics_data["loss"])
    metrics_history['accuracy'].append(metrics_data["accuracy"])
    metrics_history['precision'].append(metrics_data["precision"])
    metrics_history['recall'].append(metrics_data["recall"])
    metrics_history['f1'].append(metrics_data["f1"])
    return metrics_history


def print_metrics(data_type: str, metrics_dict: dict):
    """
    :param data_type: "Train", "Valid" or "Test"
    :param metrics_dict: dict of metrics values
    """

    loss = metrics_dict["loss"]
    accuracy = metrics_dict["accuracy"]
    precision = metrics_dict["precision"]
    recall = metrics_dict["recall"]
    f1 = metrics_dict["f1"]

    print(f"{data_type} - Loss: {loss :.4f}, Acc: {accuracy :.2f}%, "
          f"Precision: {precision :.4f}, Recall: {recall :.4f}, F1: {f1 :.4f}")


def print_epoch_summary(epoch: int, train_metrics_dict: dict, valid_metrics_dict: dict):
    print(f"\nEpoch {epoch + 1} Summary:")
    print_metrics("Train", train_metrics_dict)
    if valid_metrics_dict:
        print_metrics("Valid", valid_metrics_dict)


def setup_event_handlers(trainer, optimizer,
                         train_evaluator, valid_evaluator,
                         train_metrics_history, valid_metrics_history,
                         train_loader, valid_loader,
                         silent=False, log_interval=100):
    if not silent:
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=log_interval), log_iteration_loss)

        def log_lr():
            for param_group in optimizer.param_groups:
                print(f"Optimizer learning rate = {param_group['lr']}")
            print()

        valid_evaluator.add_event_handler(Events.COMPLETED, log_lr)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              run_evaluators_on_epoch,
                              train_evaluator, valid_evaluator,
                              train_loader, valid_loader)

    train_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                      log_and_save_epoch_results,
                                      label="Train", metrics_history=train_metrics_history, silent=silent)

    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                      log_and_save_epoch_results,
                                      label="Valid", metrics_history=valid_metrics_history, silent=silent)

    scheduler = ReduceLROnPlateauScheduler(optimizer, metric_name="loss", factor=0.5, patience=1, threshold=0.05)
    valid_evaluator.add_event_handler(Events.COMPLETED, scheduler)


def save_best_models(current_metrics, model, model_name, best_loss, best_f1, save_dir):
    save_info = []
    os.makedirs(save_dir, exist_ok=True)
    if current_metrics['loss'] < best_loss:
        best_loss = current_metrics['loss']
        loss_path = os.path.join(save_dir, f"{model_name}_best_loss.pt")
        torch.save(model.state_dict(), loss_path)
        save_info.append(f"Loss: {best_loss:.4f}")

    if current_metrics['f1'] > best_f1:
        best_f1 = current_metrics['f1']
        f1_path = os.path.join(save_dir, f'{model_name}_best_f1.pt')
        torch.save(model.state_dict(), f1_path)
        save_info.append(f"F1: {best_f1:.4f}")

    if save_info:
        print(f"Saved new best model(s) - {' | '.join(save_info)}")

    return best_loss, best_f1
