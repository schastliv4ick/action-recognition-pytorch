import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Precision, Recall, Accuracy, Fbeta, Loss
from ignite.handlers import ReduceLROnPlateauScheduler


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
