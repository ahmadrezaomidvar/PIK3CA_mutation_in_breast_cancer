"""The dataset module for the PIK3CA dataset."""

import pandas as pd
import numpy as np
from typing import Generator, Tuple, cast
from numpy.typing import NDArray
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


def train_validation_folds(p_ids, targets, n_splits=5, seed=1221):
    """
    The function takes a list of patient IDs, a list of targets, and returns k fold splits, where
    contains a list of patient IDs and targets for each fold.

    Args:
      p_ids: the patient ids
      targets: the target values for each patient
      n_splits: number of folds. Defaults to 5
      seed: The seed for the random number generator. Defaults to 1221

    Returns:
      A generator object that yields the indices of the training and validation sets for each fold.
    """
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return k_fold.split(p_ids, targets)


def get_train_validation_folds(
    root: str, n_splits: int
) -> Generator[Tuple[NDArray, NDArray, NDArray, NDArray], None, None]:
    """
    The function takes the train metadata and train outputs, and creates a generator that yields the train and
    validation folds

    Args:
      root : the path to the root of the data directory
      n_splits : number of folds
    """
    train_metadata = pd.read_csv(f"{root}/supplementary_data/train_metadata.csv")
    train_outputs = pd.read_csv(f"{root}/train_output.csv")
    train_info = train_metadata.merge(train_outputs, on="Sample ID")
    _p_ids = train_info["Patient ID"]

    train_info_unique = train_info.drop_duplicates(subset="Patient ID", keep="first")
    p_ids = np.unique(_p_ids)
    targets = train_info_unique[train_info_unique["Patient ID"].isin(p_ids)][
        "Target"
    ].values

    folds = train_validation_folds(p_ids=p_ids, targets=targets, n_splits=n_splits)
    for train_index_, validation_index_ in folds:
        train_index = np.arange(len(train_info))[
            pd.Series(_p_ids).isin(p_ids[train_index_])
        ]
        validation_index = np.arange(len(train_info))[
            pd.Series(_p_ids).isin(p_ids[validation_index_])
        ]
        train = train_info.iloc[train_index]
        validation = train_info.iloc[validation_index]
        train_x = train["Sample ID"].values
        train_y = train["Target"].values

        validation_x = validation["Sample ID"].values
        validation_y = validation["Target"].values
        yield cast(NDArray, train_x), cast(NDArray, train_y), cast(
            NDArray, validation_x
        ), cast(NDArray, validation_y)


class PIK3CAData(Dataset):
    """The PIK3CA dataset for training and validation."""

    def __init__(
        self, sample_ids: NDArray, targets: NDArray, root: str, type: str
    ) -> None:
        """
        Initialize the dataset

        Args:
          sample_ids: a list of sample ids (strings)
          targets: The targets for the dataset.
          root: the path to the root directory of the dataset
          type: the type of the dataset. Either train or test

        Raises:
          ValueError: If the type is not train or test
        """
        super().__init__()
        self.type = type
        self.path_to_data = f"{root}/{self.type}_input/moco_features"

        if self.type == "train":
            self.sample_ids = sample_ids
            self.targets = targets
            assert len(self.sample_ids) == len(
                self.targets
            ), "sample_ids and targets must have the same length"

        elif self.type == "test":
            test_metadata = pd.read_csv(f"{root}/supplementary_data/test_metadata.csv")
            test_x = test_metadata["Sample ID"].values
            self.sample_ids = cast(NDArray, test_x)

        else:
            raise ValueError(f"Invalid type: {self.type}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Tuple[NDArray, int]:
        """
        The method takes in an index, and returns a tuple of the data and the label

        Args:
          index: the index of the sample to be retrieved

        Returns:
          The x is the data and the y is the target.
        """

        x = np.load(f"{self.path_to_data}/{self.sample_ids[index]}")
        x = np.swapaxes(x[:, 3:], 0, 1)
        if self.type == "train":
            y = self.targets[index]
            return x, y

        else:
            return x, self.sample_ids[index]
