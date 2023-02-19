"""Train the model."""

import torch
from chowder.utils.engine import train_one_epoch, evaluate
import yaml
from chowder.utils.utils import (
    get_device,
    make_model,
    make_loss,
    make_optimizer,
    make_dataset,
)
from chowder.constant import PACKAGE_ROOT_PATH
from pathlib import Path
from typing import Tuple, Dict, List
from chowder.dataset import get_train_validation_folds
from datetime import datetime
import logging

# Loading the config file.
config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer(object):
    """Trainer class for training the model."""

    def __init__(self, config: Dict, generation_fold: Tuple) -> None:
        """
        Initialize the trainer class.
        The function takes in a dictionary of configuration parameters and a tuple of training and
        validation data, and returns a model, optimizer, loss function, and data loaders

        Args:
          config: A dictionary containing the hyperparameters for the model.
          generation_fold (Tuple): The fold of the data that we're using for training.
        """
        super().__init__()
        self.config = config
        self.device = get_device()
        self.model = make_model(config=self.config).to(self.device)
        self.optimizer = make_optimizer(self.model, lr=self.config["lr"])
        self.loss = make_loss()
        self.data_loader_train, self.data_loader_val = make_dataset(
            generation_fold=generation_fold, config=self.config, type="train"
        )

    def train(self, num_epochs: int) -> Tuple[Dict, float]:
        """
        Train method to train the model for a number of epochs, and at the end of each epoch,
        it evaluates the model on the validation set. If the model performs better than the
        previous best model, it saves the model

        Args:
          num_epochs: Number of epochs to train for

        Returns:
          The best model and the best AUC
        """
        start_epoch = 0
        counter = 0
        best_auc = 0
        best_model: Dict = {}

        for epoch in range(start_epoch, num_epochs + start_epoch):
            metric_logger, counter = train_one_epoch(
                self.model,
                self.loss,
                self.optimizer,
                self.data_loader_train,
                self.device,
                epoch,
                counter=counter,
            )
            logger.info(
                f'Epoch: {epoch} Train Loss: {metric_logger.meters["loss"].global_avg:.4f}'
            )

            val_auc = evaluate(
                self.model,
                self.loss,
                self.data_loader_val,
                device=self.device,
                epoch=epoch,
            )
            logger.info(f"Epoch: {epoch} Validation AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                logger.info(f"Model saved with AUC: {best_auc:.4f}")
                best_model = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                }

        return best_model, best_auc


def k_fold_train(config: Dict) -> None:
    """
    The function to handle the training of the model for k folds. The function takes in a configuration
    dictionary and returns a list of the best models and a list of the best aucs for each fold.

    Args:
      config: Dict
    """
    root = config["root"]
    num_epochs = config["num_epochs"]
    n_splits = config["n_splits"]
    best_models: List[Dict] = []
    best_aucs: List[float] = []
    train_validation_generator = get_train_validation_folds(
        root=root, n_splits=n_splits
    )
    for i, dataset in enumerate(train_validation_generator):
        logger.info(f"Fold {i+1} of {n_splits}")
        trainer = Trainer(config, dataset)
        best_model, best_auc = trainer.train(num_epochs=num_epochs)
        best_models.append(best_model)
        best_aucs.append(best_auc)

    to_save = Path(root) / "checkpoints"
    to_save.mkdir(parents=True, exist_ok=True)
    torch.save(
        best_models,
        to_save / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_best_models.pth',
    )
    torch.save(
        best_aucs,
        to_save
        / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_best_aucs_{config["reduce_method"]}.pth',
    )


if __name__ == "__main__":
    k_fold_train(config=config)
