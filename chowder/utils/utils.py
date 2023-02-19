"""Utility functions for training and evaluating models."""

import torch
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import logging
import time
from chowder.model import Chowder
from chowder.dataset import PIK3CAData
from typing import Dict, Tuple, Optional
from torch import nn
from torch.utils.data import DataLoader

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # if not is_dist_avail_and_initialized():
        #     return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {}".format(header, total_time_str))


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


def get_device() -> torch.device:
    """
    If a GPU is available, use it. Otherwise, use the CPU

    Returns:
      A torch.device object.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"\nDevice is on {device} . . .\n")

    return device


def make_model(config: Dict) -> nn.Module:
    """
    The function to make the model.

    Args:
      config: Dict

    Returns:
      The model is being returned.
    """
    model = Chowder(
        features_dim=config["features_dim"],
        n_kernels=config["n_kernels"],
        quantiles=config["quantiles"],
        retained_features=config["retained_features"],
        n_first_mlp_neurons=config["n_first_mlp_neurons"],
        n_second_mlp_neurons=config["n_second_mlp_neurons"],
        reduce_method=config["reduce_method"],
        dropout=config["dropout"],
    )
    return model


def make_loss() -> nn.Module:
    """
    The function to make the loss function.

    Returns:
      A CrossEntropyLoss object.
    """
    return nn.CrossEntropyLoss(reduction="mean")


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    > The function to make the optimizer.

    Args:
        model: The model.
        lr: The learning rate.

    Returns:
        The optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)

    return optimizer


def make_dataset(
    config: Dict, type: str, generation_fold: Optional[Tuple] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    This function takes in a tuple of training and validation data, and returns a tuple of
    training and validation dataloaders

    Args:

      config: Dict
      type: The type of dataset to be created. Either "train" or "test".
      generation_fold: Tuple of training and validation data. Only required for training.

    Returns:
        A tuple of training and validation dataloaders for training, and a test dataloader for testing.
    """
    if type == "train":
        if not generation_fold:
            raise ValueError("generation_fold must be provided for training")
        train_x, train_y, validation_x, validation_y = generation_fold

        train_dataset = PIK3CAData(
            sample_ids=train_x, targets=train_y, root=config["root"], type="train"
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        validation_dataset = PIK3CAData(
            sample_ids=validation_x,
            targets=validation_y,
            root=config["root"],
            type="train",
        )
        # batch size is the entire validation set for validation to calculate auc score over the entire set
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=validation_dataset.__len__(),
            shuffle=True,
            num_workers=4,
        )

        return train_loader, validation_loader

    elif type == "test":
        test_dataset = PIK3CAData(
            sample_ids=None, targets=None, root=config["root"], type="test"
        )
        # batch size is the entire test set for testing to calculate auc score over the entire set
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_dataset.__len__(),
            shuffle=False,
            num_workers=4,
        )

        return test_loader
