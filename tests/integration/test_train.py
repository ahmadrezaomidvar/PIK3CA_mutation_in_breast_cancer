"""Test the integration of the package."""

from chowder.train import k_fold_train as train_main
from chowder.constant import PACKAGE_ROOT_PATH


class TestIntegrationTrain:
    """
    Test the integration of the train.py module
    """

    def setup_method(self):
        """
        Setup the config for the test
        """
        root = PACKAGE_ROOT_PATH.parent / "tests" / "fixture" / "data"
        self.config = {
            "root": str(root),
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
            "l2_reg": 0.5,
            "dropout": 0.5,
            "n_ensemble": 1,
            "mean": 0.0418,
            "std": 0.116,
        }

    def test_integration_train(self):
        """
        Test the integration of the train.py module
        """
        train_main(config=self.config)
