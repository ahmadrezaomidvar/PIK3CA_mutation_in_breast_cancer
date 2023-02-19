import torch
from torch import nn
from chowder.utils.utils import get_device, Logger
from chowder.model import Chowder
from chowder.dataset import PIK3CAData_test
from chowder.dataset import PIK3CAData_test
from chowder.constant import PACKAGE_ROOT_PATH
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import yaml

# Loading the config file.
config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
logger = Logger(log_dir=Path(config["root"]).joinpath("log")).make_logger()


class Predictor(object):
    """Predictor class for predicting the model."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize the predictor class.
        The function takes in a dictionary of configuration parameters and returns a model and data loader

        Args:
          config (Dict): A dictionary containing the hyperparameters for the model.
        """
        super().__init__()
        self.config = config
        self.device = get_device()
        self.model = self.make_model(config=self.config)
        self.data_loader = self.make_dataset()
        self.checkpoint_path = Path(self.config["root"]).joinpath(
            "checkpoints", self.config["checkpoint_name"]
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

        return model

    def make_dataset(self) -> DataLoader:
        """
        The function to make the test dataset.

        Args:
            None

        Returns:
          The test dataloader is being returned.
        """

        test_dataset = PIK3CAData_test(root=self.config["root"])
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_dataset.__len__(),
            shuffle=False,
            num_workers=4,
        )

        return test_loader

    def predict(self, root: str) -> pd.DataFrame:
        """
        The function to make the predictions.

        Args:
          root (str): the path to the folder where the test images are stored.

        Returns:
            A pandas dataframe containing the predictions.

        """

        checkpoints = torch.load(self.checkpoint_path, map_location="cpu")
        pred = 0
        for checkpoint in checkpoints:
            self.model.load_state_dict(checkpoint["model"])

            with torch.no_grad():
                self.model.eval()
                to_save = Path(root).joinpath("predictions")
                to_save.mkdir(parents=True, exist_ok=True)

                for x, test_id in tqdm(iter(self.data_loader)):
                    x = x.to(self.device)
                    output = self.model(x)
                    output_proba = nn.Softmax(dim=1)(output)
                    pred += output_proba[:, 1].cpu().numpy()

        pred = pred / len(checkpoints)

        submission = pd.DataFrame({"Sample ID": test_id, "Target": pred}).sort_values(
            "Sample ID"
        )

        # sanity checks
        assert all(
            submission["Target"].between(0, 1)
        ), "`Target` values must be in [0, 1]"
        assert submission.shape == (
            149,
            2,
        ), "Your submission file must be of shape (149, 2)"
        assert list(submission.columns) == [
            "Sample ID",
            "Target",
        ], "Your submission file must have the following columns: `Sample ID`, `Target`"

        logger.info("Predictions saved.\n")

        return submission


if __name__ == "__main__":
    to_save = Path(config["root"]).joinpath("predictions")
    to_save.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(config=config)
    submission = predictor.predict(root=config["root"])

    # Saving the predictions to a csv file.
    submission.to_csv(
        to_save.joinpath(
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_benchmark_test_output.csv'
        ),
        index=False,
    )
