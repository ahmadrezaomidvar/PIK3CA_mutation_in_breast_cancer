"""Test the dataset."""

from chowder.dataset import get_train_validation_folds, PIK3CAData
from chowder.constant import PACKAGE_ROOT_PATH


class TestDataset:
    """Test the dataset."""

    def setup_method(self):
        """Setup the config for the test."""

        self.root = PACKAGE_ROOT_PATH.parent / "data"
        self.config = {
            "root": str(self.root),
            "reduce_method": "minmax",
            "n_kernels": 1,
            "retained_features": 2,
            "quantiles": [0.25, 0.75],
            "n_first_mlp_neurons": 200,
            "n_second_mlp_neurons": 100,
            "features_dim": 2048,
            "seed": 1221,
            "n_splits": 2,
            "batch_size": 4,
            "lr": 0.001,
            "num_epochs": 1,
        }

    def test_pik3cadata_train(self):
        """Test the PIK3CAData class."""
        train_validation_generator = get_train_validation_folds(
            root=self.root, n_splits=2
        )
        train_x, train_y, _, _ = next(train_validation_generator)

        training_dataset = PIK3CAData(
            sample_ids=train_x,
            targets=train_y,
            root=self.root,
            type="train",
        )
        _ = training_dataset.__getitem__(0)
        _ = training_dataset.__len__()

    def test_pik3cadata_test(self):
        """Test the PIK3CAData class."""
        train_validation_generator = get_train_validation_folds(
            root=self.root, n_splits=2
        )
        train_x, train_y, _, _ = next(train_validation_generator)

        training_dataset = PIK3CAData(
            sample_ids=train_x,
            targets=train_y,
            root=self.root,
            type="test",
        )
        _ = training_dataset.__getitem__(0)
        _ = training_dataset.__len__()
