import torch.nn as nn
import torch
from torch.autograd import Function

class Chowder(nn.Module):
    
    def __init__(self, features_dim: int, J: int, R: int, n_first_mlp_neurons: int=200, n_second_mlp_neurons: int=100) -> None:
        super().__init__()
        
        self.embedding_layer = nn.Conv1d(features_dim, J, 1)
        self.min_max_layer = MinMaxModule(k=R)

        self.mlp = nn.Sequential(
            nn.Linear(2 * R * J, n_first_mlp_neurons),
            nn.Sigmoid(),
            nn.Linear(n_first_mlp_neurons, n_second_mlp_neurons),
            nn.Sigmoid(),
            nn.Linear(n_second_mlp_neurons, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.embedding_layer(x)
        x = self.min_max_layer(x)
        x = x.view(x.size(0), -1)
        logits = self.mlp(x)

        return logits

class MinMaxModule(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k


    def forward(self, input: torch.Tensor)-> torch.Tensor:

        top_max_k, _ = torch.topk(input, dim=2, k=self.k)

        input_negative = torch.negative(input)
        top_min_k_neg, _ = torch.topk(input_negative, dim=2, k=self.k)
        top_min_k = torch.negative(top_min_k_neg)

        output = torch.cat((top_max_k,top_min_k), dim=2)
        output,_ = torch.sort(output, dim=2, descending=True)

        return output

## TO DO
#
#

if __name__ == '__main__':
    

    model = Chowder(2048, J=1, R=2, n_first_mlp_neurons=200, n_second_mlp_neurons=100)

    from dataset import DataFeatures
    root = "/Users/rezachi/ML/datasets/owkin/data/"
    data = DataFeatures(root)
    X_train, y_train, centers_train, patients_train = data.__getdata__()

    x = model(X_train)
    print(x.shape)
    print(x[54])

    print('\n    Total params: %.2f No' % (sum(p.numel() for p in model.parameters())))
    print('    Total trainable params: %.0f No' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
