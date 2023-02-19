import torch
import torch.nn as nn
from utils.engine import train_one_epoch, evaluate
import yaml
from utils.utils import get_device
from model import Chowder
from dataset import PIK3CAData
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, List
from dataset import get_train_validation_folds
import time
from datetime import datetime

config_path = './configs/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

class Trainer(object):

    def __init__(self, config: Dict, generation_fold: Tuple) -> None:
        super().__init__()
        self.config = config
        self.device = get_device()
        self.model = self.make_model(config=self.config)
        self.optimizer = self.make_optimizer(lr = self.config['lr'])
        self.loss = self.make_loss()
        self.data_loader_train, self.data_loader_val = self.make_dataset(generation_fold)

    def make_model(self, config: Dict) -> nn.Module:
        model = Chowder(features_dim=config['features_dim'], J=config['J'], R=config['R'], n_first_mlp_neurons=config['n_first_mlp_neurons'], n_second_mlp_neurons=config['n_second_mlp_neurons'])
        model.to(self.device)
        print('\n    Total params: %.2f No' % (sum(p.numel() for p in model.parameters())))
        print('    Total trainable params: %.0f No' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

        return model

    def make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)

        return optimizer

    def make_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss(reduction='mean')

    def make_dataset(self, generator_fold: Tuple) -> Tuple[DataLoader, DataLoader]:
        train_x, train_y, validation_x, validation_y = generator_fold

        train_dataset = PIK3CAData(
            sample_ids=train_x,
            targets=train_y,
            root=self.config['root'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

        validation_dataset = PIK3CAData(
            sample_ids=validation_x,
            targets=validation_y,
            root=self.config['root'])
        validation_loader = DataLoader(validation_dataset, batch_size=validation_dataset.__len__(), shuffle=True, num_workers=4)

        return train_loader, validation_loader

    def train(self, num_epochs: int) -> Dict:
        start_epoch = 0
        counter = 0
        best_auc = 0
        best_model = None

        for epoch in range(start_epoch, num_epochs + start_epoch):
            _, counter = train_one_epoch(self.model, self.loss, self.optimizer, self.data_loader_train, self.device, epoch, 
            counter=counter,
            )

            val_auc = evaluate( self.model, self.loss, self.data_loader_val, device=self.device, epoch=epoch)
            
            if val_auc > best_auc:
                best_auc = val_auc
                print(f'Model saved with AUC: {best_auc:.4f}')
                best_model = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                    }

        return best_model, best_auc

def k_fold_train(root: str, num_epochs: int, k:int) -> Tuple[List[Dict], List[float]]:
    best_models = []
    best_aucs = []
    train_validation_generator = get_train_validation_folds(root=root, n_splits=k)
    for i in range(k):
        print(f'Fold {i+1} of {k}')
        dataset = next(train_validation_generator)
        trainer = Trainer(config, dataset)
        best_model, best_auc = trainer.train(num_epochs=num_epochs)
        best_models.append(best_model)
        best_aucs.append(best_auc)

    to_save = Path(root).joinpath('checkpoints')
    to_save.mkdir(parents=True, exist_ok=True)
    torch.save(best_models, to_save.joinpath(f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_best_models.pth'))
    torch.save(best_aucs, to_save.joinpath(f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_best_aucs.pth'))


if __name__ == "__main__":
    k_fold_train(root= config['root'] ,num_epochs=config['num_epochs'], k=config['n_splits'])
