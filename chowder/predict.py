"""Predictor class for predicting the model."""

import torch
from torch import nn
from chowder.utils.utils import get_device, make_model, make_dataset
from chowder.constant import PACKAGE_ROOT_PATH
from pathlib import Path
from typing import Dict
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import yaml
import logging

# Loading the config file.
config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Predictor(object):
    """Predictor class for predicting the model."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize the predictor class.
        The function takes in a dictionary of configuration parameters and returns a model and data loader

        Args:
          config: A dictionary containing the hyperparameters for the model.
        """
        super().__init__()
        self.config = config
        self.device = get_device()
        self.model = make_model(config=self.config).to(self.device)
        self.data_loader = make_dataset(config=self.config, type="test")
        self.checkpoint_path = Path(self.config["root"]).joinpath(
            "checkpoints", self.config["checkpoint_name"]
        )

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
        for checkpoint in tqdm(checkpoints):
            self.model.load_state_dict(checkpoint["model"])

            with torch.no_grad():
                self.model.eval()
                to_save = Path(root) / "predictions"
                to_save.mkdir(parents=True, exist_ok=True)

                x, test_id = next(iter(self.data_loader))
                x = x.to(self.device)
                output = self.model(x)
                output_proba = nn.Softmax(dim=1)(output)
                # as the entire dataset is loaded into memory for prediction, we can calculate the pred TODO add support for large datasets
                pred += output_proba[:, 1].cpu().numpy()

        pred = pred / len(checkpoints)

        # as the entire dataset is loaded into memory for prediction, we can use test_id TODO add support for large datasets
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
    to_save = Path(config["root"]) / "predictions"
    to_save.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(config=config)
    submission = predictor.predict(root=config["root"])

    # Saving the predictions to a csv file.
    submission.to_csv(
        to_save
        / f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_benchmark_test_output.csv',
        index=False,
    )
