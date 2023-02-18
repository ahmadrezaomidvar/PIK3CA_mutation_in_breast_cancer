from pathlib import Path
# from PIL import Image
# import os
import pandas as pd
from tqdm import tqdm
import numpy as np
# import glob

import torch
from torch.utils.data import Dataset, DataLoader
# import torchvision


# Dataset
class DataFeatures(Dataset):
    
    def __init__(self, root):
        super(DataFeatures, self).__init__()
        
        # parameters
        self.root = Path(root)
        self.train_features_dir = self.root.joinpath('train_input', 'moco_features')
        self.test_features_dir  = self.root.joinpath('test_input', 'moco_features')
        self.df_train = pd.read_csv(self.root.joinpath('supplementary_data', 'train_metadata.csv'))
        self.df_test  = pd.read_csv(self.root.joinpath('supplementary_data', 'test_metadata.csv'))
        self.y_train = pd.read_csv(self.root.joinpath('train_output.csv'))

        self.X_train_df, self.y_train_df, self.centers_train_df, self.patients_train_df = self.make_train()    


    def make_train(self):

        self.df_train = self.df_train.merge(self.y_train, on='Sample ID')
        X_train = []
        y_train = []
        centers_train = []
        patients_train = []

        for sample, label, center, patient in tqdm(
            self.df_train[["Sample ID", "Target", "Center ID", "Patient ID"]].values
            ):
            _features = np.load(self.train_features_dir / sample)
            coordinates, features = _features[:, :3], _features[:, 3:]

            X_train.append(features)
            y_train.append(label)
            centers_train.append(center)
            patients_train.append(patient)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        centers_train = np.array(centers_train)
        patients_train = np.array(patients_train)

        self.n_samples = X_train.shape[0]

        return X_train, y_train, centers_train, patients_train

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        X_train = torch.from_numpy(self.X_train_df[index]).float().permute(1, 0)
        y_train = torch.Tensor([self.y_train_df[index]]).long()
        centers_train = self.centers_train_df[index]
        patients_train = self.patients_train_df[index]

        return (X_train, y_train, centers_train, patients_train)

    def __getdata__(self):

        X_train = torch.from_numpy(self.X_train_df).float().permute(0, 2, 1)
        y_train = torch.from_numpy(self.y_train_df).long()
        centers_train = self.centers_train_df
        patients_train = self.patients_train_df

        return (X_train, y_train, centers_train, patients_train)

# TODO

if __name__ == '__main__':
    root = "/Users/rezachi/ML/datasets/owkin/data/"
    data = DataFeatures(root)
    X_train, y_train, centers_train, patients_train = data.__getitem__(0)
    print(X_train.shape)
    print(y_train.shape)
    print(centers_train.shape)
    print(patients_train.shape)
    print(data.__len__())

    X_train, y_train, centers_train, patients_train = data.__getdata__()
    print(X_train.shape)
    print(y_train.shape)
    print(centers_train.shape)
    print(patients_train.shape)
    print(data.__len__())
