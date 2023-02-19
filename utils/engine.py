import math
import sys
import torch
from torch import nn

from utils.utils import MetricLogger, SmoothedValue
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=10, log_dir='.', counter=0):#, _run = None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for x, y in metric_logger.log_every(data_loader, print_freq, header):
        counter = counter +1
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            print(loss)
            sys.exit(1)
    
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return metric_logger,counter


def evaluate(model, criterion, data_loader, device, epoch, print_freq=10, log_dir='.'):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for x, y in metric_logger.log_every(data_loader, print_freq, header):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            output = model(x)
            loss = criterion(output, y)
            output_proba = nn.Softmax(dim=1)(output)
            auc = roc_auc_score(y.cpu().numpy(), output_proba[:, 1].cpu().numpy())
            metric_logger.update(loss=loss.item())
            metric_logger.meters['auc'].update(auc.item())

    return metric_logger.auc.global_avg