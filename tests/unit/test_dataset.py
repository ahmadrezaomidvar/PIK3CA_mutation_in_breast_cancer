from chowder.dataset import get_train_validation_folds, PIK3CAData
from chowder.constant import PACKAGE_ROOT_PATH
from pathlib import Path
import yaml


class TestDataset:
    """Test the dataset."""

    def setup_method(self):
        """Setup the config for the test."""
        config_path = PACKAGE_ROOT_PATH / "configs/config.yaml"
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
        self.root = self.config["root"]

    def test_pik3cadata(self):
        """Test the PIK3CAData class."""
        train_validation_generator = get_train_validation_folds(
            root=self.root, n_splits=2
        )
        train_x, train_y, _, _ = next(train_validation_generator)

        training_dataset = PIK3CAData(
            sample_ids=train_x,
            targets=train_y,
            root=self.root,
        )
        _ = training_dataset.__getitem__(0)
        _ = training_dataset.__len__()
