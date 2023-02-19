import torch
import torch.nn as nn
from chowder.utils.engine import train_one_epoch, evaluate
import yaml
from chowder.utils.utils import get_device
from chowder.model import Chowder
from chowder.dataset import PIK3CAData
from chowder.constant import PACKAGE_ROOT_PATH
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, List
from chowder.dataset import get_train_validation_folds
from datetime import datetime
from chowder.utils.utils import Logger

# Loading the config file.
config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

logger = Logger(log_dir=Path(config["root"]).joinpath("log")).make_logger()


class Trainer(object):
    """Trainer class for training the model."""

    def __init__(self, config: Dict, generation_fold: Tuple) -> None:
        """
        Initialize the trainer class.
        The function takes in a dictionary of configuration parameters and a tuple of training and
        validation data, and returns a model, optimizer, loss function, and data loaders

        Args:
          config (Dict): A dictionary containing the hyperparameters for the model.
          generation_fold (Tuple): The fold of the data that we're using for training.
        """
        super().__init__()
        self.config = config
        self.device = get_device()
        self.model = self.make_model(config=self.config)
        self.optimizer = self.make_optimizer(lr=self.config["lr"])
        self.loss = self.make_loss()
        self.data_loader_train, self.data_loader_val = self.make_dataset(
            generation_fold
        )

    def make_model(self, config: Dict) -> nn.Module:
        """
        The function to make the model.

        Args:
          config (Dict): Dict

        Returns:
          The model is being returned.
        """
        model = Chowder(
            features_dim=config["features_dim"],
            J=config["J"],
            R=config["R"],
            n_first_mlp_neurons=config["n_first_mlp_neurons"],
            n_second_mlp_neurons=config["n_second_mlp_neurons"],
        )
        model.to(self.device)
        print("Total params: %.2f No" % (sum(p.numel() for p in model.parameters())))
        print(
            "Total trainable params: %.0f No"
            % (sum(p.numel() for p in model.parameters() if p.requires_grad))
        )

        return model

    def make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """
        > The function to make the optimizer.

        Args:
          lr (float): The learning rate.

        Returns:
          The optimizer
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)

        return optimizer

    def make_loss(self) -> nn.Module:
        """
        The function to make the loss function.

        Returns:
          A CrossEntropyLoss object.
        """
        return nn.CrossEntropyLoss(reduction="mean")

    def make_dataset(self, generator_fold: Tuple) -> Tuple[DataLoader, DataLoader]:
        """
        This function takes in a tuple of training and validation data, and returns a tuple of
        training and validation dataloaders

        Args:
          generator_fold (Tuple): Tuple

        Returns:
          A tuple of two dataloaders, one for training and one for validation.
        """
        train_x, train_y, validation_x, validation_y = generator_fold

        train_dataset = PIK3CAData(
            sample_ids=train_x, targets=train_y, root=self.config["root"]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        validation_dataset = PIK3CAData(
            sample_ids=validation_x, targets=validation_y, root=self.config["root"]
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=validation_dataset.__len__(),
            shuffle=True,
            num_workers=4,
        )

        return train_loader, validation_loader

    def train(self, num_epochs: int) -> Dict:
        """
        Train method to train the model for a number of epochs, and at the end of each epoch, 
        it evaluates the model on the validation set. If the model performs better than the 
        previous best model, it saves the model

        Args:
          num_epochs (int): Number of epochs to train for

        Returns:
          The best model and the best AUC
        """
        start_epoch = 0
        counter = 0
        best_auc = 0
        best_model = None

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


def k_fold_train(config: Dict) -> Tuple[List[Dict], List[float]]:
    """
    The function to handle the training of the model for k folds. The function takes in a configuration
    dictionary and returns a list of the best models and a list of the best aucs for each fold.

    Args:
      config (Dict): Dict
    """
    root = config["root"]
    num_epochs = config["num_epochs"]
    k = config["n_splits"]
    best_models = []
    best_aucs = []
    train_validation_generator = get_train_validation_folds(root=root, n_splits=k)
    for i in range(k):
        logger.info(f"Fold {i+1} of {k}")
        dataset = next(train_validation_generator)
        trainer = Trainer(config, dataset)
        best_model, best_auc = trainer.train(num_epochs=num_epochs)
        best_models.append(best_model)
        best_aucs.append(best_auc)

    to_save = Path(root).joinpath("checkpoints")
    to_save.mkdir(parents=True, exist_ok=True)
    torch.save(
        best_models,
        to_save.joinpath(
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_best_models.pth'
        ),
    )
    torch.save(
        best_aucs,
        to_save.joinpath(
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_best_aucs.pth'
        ),
    )


if __name__ == "__main__":
    k_fold_train(config=config)
