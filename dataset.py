import pandas as pd
import numpy as np
from typing import Generator, Tuple, cast
from numpy.typing import NDArray
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


def train_validation_folds(p_ids, targets, n_splits=5, seed=1221):
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return k_fold.split(p_ids, targets)

def get_train_validation_folds(root: str, n_splits:int) -> (
    Generator[Tuple[NDArray, NDArray, NDArray, NDArray], None, None]
):
    train_metadata = pd.read_csv(f'{root}/supplementary_data/train_metadata.csv')
    train_outputs = pd.read_csv(f'{root}/train_output.csv')
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
    
    def __init__(self, sample_ids: NDArray, targets: NDArray, root: str) -> None:
        super().__init__()
        
        self.sample_ids = sample_ids
        self.targets = targets
        self.path_to_data = f'{root}/train_input/moco_features'
        assert len(self.sample_ids) == len(self.targets), "sample_ids and targets must have the same length"

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Tuple[NDArray, int]:

        x = np.load(f'{self.path_to_data}/{self.sample_ids[index]}')
        x = np.swapaxes(x[:,3:],0,1)
        y = self.targets[index]

        return x, y

class PIK3CAData_test(Dataset):
    
    def __init__(self, root: str) -> None:
        super().__init__()
        
        self.path_to_data = f'{root}/test_input/moco_features'
        test_metadata = pd.read_csv(f'{root}/supplementary_data/test_metadata.csv')
        test_x = test_metadata["Sample ID"].values
        self.sample_ids = cast(NDArray, test_x)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Tuple[NDArray, str]:

        test_id = self.sample_ids[index]
        x = np.load(f'{self.path_to_data}/{test_id}')
        x = np.swapaxes(x[:,3:],0,1)

        return x, test_id