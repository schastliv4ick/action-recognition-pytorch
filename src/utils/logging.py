from torch import Tensor
from ignite.engine import Events
from ignite.handlers import ReduceLROnPlateauScheduler

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
