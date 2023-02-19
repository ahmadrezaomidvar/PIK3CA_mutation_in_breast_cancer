import torch
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import logging
from pathlib import Path
import time


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
        if not is_dist_avail_and_initialized():
            return
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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_device() -> torch.device:
    """
    If a GPU is available, use it. Otherwise, use the CPU

    Returns:
      A torch.device object.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\nDevice is on {device} . . .\n")

    return device


class Logger(object):
    """
    logger preparation


    Parameters
    ----------
    log_dir: string
        path to the log directory

    logging_level: string
        required Level of logging. INFO, WARNING or ERROR can be selected. Default to 'INFO'

    console_logger: bool
        flag if console_logger is required. Default to False

    Returns
    ----------
    logger: logging.Logger
        logger object
    """

    def __init__(
        self, log_dir, logging_level="INFO", console_logger=True, multi_module=True
    ) -> None:
        super().__init__()
        self._log_dir = log_dir
        self.console_logger = console_logger
        self.logging_level = logging_level.lower()
        self.multi_module = multi_module
        self._make_level()

    def _make_level(self):
        if self.logging_level == "info":
            self._level = logging.INFO
        elif self.logging_level == "warning":
            self._level = logging.WARNING
        elif self.logging_level == "error":
            self._level = logging.ERROR
        else:
            raise ValueError(
                "logging_level not specified correctly. INFO, WARNING or ERROR must be chosen"
            )

    def make_logger(self):
        # logging configuration
        log_dir = Path(self._log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_name = log_dir.joinpath(f'{time.strftime("%Y%m%d")}.log')

        # Create a custom logger
        if self.multi_module:
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(__name__)
        logger.setLevel(self._level)

        # Create handlers
        f_handler = logging.FileHandler(filename=file_name)
        f_handler.setLevel(self._level)

        # Create formatters
        format = logging.Formatter(
            "%(name)s - %(asctime)s - %(levelname)s - %(message)s"
        )
        f_handler.setFormatter(format)

        # Add handlers to the logger
        logger.addHandler(f_handler)

        # Console handler creation
        if self.console_logger:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(self._level)
            c_handler.setFormatter(format)
            logger.addHandler(c_handler)

        return logger
