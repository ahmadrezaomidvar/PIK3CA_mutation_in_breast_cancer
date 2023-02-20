"""This module contains the training and evaluation functions for the model."""

import math
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple

from chowder.utils.utils import MetricLogger, SmoothedValue
from sklearn.metrics import roc_auc_score

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
    counter: int = 0,
    l2_reg: float = 0.0,
) -> Tuple[MetricLogger, int]:
    """
    The function trains the model for one epoch

    Args:
      model: the model we're training
      criterion: The loss function.
      optimizer: The optimizer to use.
      data_loader: the dataloader for the training set
      device: the device to train on.
      epoch: The current epoch number.
      print_freq: How often to print the loss. Defaults to 10
      counter: This is a counter that keeps track of the number of batches that have been trained.
      l2_reg: The L2 regularization parameter. Defaults to 0
    Defaults to 0

    Returns:
      The metric_logger and the counter
    """

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for x, y in metric_logger.log_every(data_loader, print_freq, header):
        counter = counter + 1
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        # Add L2 regularization to the embedding layer as described in the paper
        l2_regularisation = torch.tensor(0.0).to(device)
        embedding_weight = model.embedding_layer.weight
        l2_regularisation += torch.norm(embedding_weight)
        loss += l2_reg * l2_regularisation

        loss.backward()
        optimizer.step()

        if not math.isfinite(loss):
            raise ValueError("Loss is {loss}, stopping training")

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return metric_logger, counter


def evaluate(
    model: nn.modules,
    criterion: nn.modules,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
) -> float:
    """
    The function evaluates the model on the validation set.

    Args:
      model: the model to train
      criterion: the loss function
      data_loader: the data loader for the test set
      device: the device to run the training on.
      epoch: the current epoch number
      print_freq: how often to print the loss and auc. Defaults to 10
      log_dir: the directory where the logs will be saved. Defaults to .

    Returns:
      The average AUC over the entire test set.
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        for x, y in metric_logger.log_every(data_loader, print_freq, header):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            output_proba = nn.Softmax(dim=1)(output)
            # as the entire dataset is loaded into memory for evaluation, we can calculate the AUC TODO add support for large datasets
            auc = roc_auc_score(y.cpu().numpy(), output_proba[:, 1].cpu().numpy())
            metric_logger.update(loss=loss.item())
            metric_logger.meters["auc"].update(auc.item())

    return metric_logger.auc.global_avg
