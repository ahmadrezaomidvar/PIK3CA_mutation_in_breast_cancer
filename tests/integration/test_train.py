from chowder.train import k_fold_train as train_main


class TestIntegrationTrain:
    """
    Test the integration of the train.py module
    """

    def setup_method(self):
        """
        Setup the config for the test
        """
        self.config = {
            "root": "/Users/rezachi/ML/datasets/owkin/data",
            "J": 1,
            "R": 2,
            "n_first_mlp_neurons": 200,
            "n_second_mlp_neurons": 100,
            "features_dim": 2048,
            "seed": 1221,
            "n_splits": 2,
            "batch_size": 4,
            "lr": 0.001,
            "num_epochs": 1,
        }

    def test_integration_train(self):
        """
        Test the integration of the train.py module
        """
        train_main(config=self.config)
