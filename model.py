import torch.nn as nn
import torch
from torch.autograd import Function

class GetModel(nn.Module):
    
    def __init__(self, features_dim, J, R, n_first_mlp_neurons=200, n_second_mlp_neurons=100):
        super(GetModel, self).__init__()
        
        self.embedding_layer = nn.Conv1d(features_dim, J, 1)
        self.min_max_layer = MinMax(R)

        self.mlp = nn.Sequential(
            nn.Linear(2*R*J, n_first_mlp_neurons),
            nn.Sigmoid(),
            nn.Linear(n_first_mlp_neurons, n_second_mlp_neurons),
            nn.Sigmoid(),
            nn.Linear(n_second_mlp_neurons, 2),
        )

    def forward(self, x):
        
        x = self.embedding_layer(x)
        x = self.min_max_layer(x)
        x = x.view(x.size(0), -1)
        logits = self.mlp(x)

        return logits



class MinMaxFunction(Function):

    @staticmethod
    def forward(ctx, input, R):
        ctx.save_for_backward(input)
        
        sorted, indices = torch.sort(input, dim=2, descending=True)
        ctx.indices_max = indices.narrow(2, 0, R)
        output = sorted.narrow(2, 0, R)

        ctx.indices_min = indices.narrow(2, -R, R)
        output = torch.cat((output, sorted.narrow(2, -R, R)), dim=2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
            
            input, = ctx.saved_tensors
            R = ctx.indices_max.size(2)

            grad_output_max = grad_output.narrow(2, 0, R)
            grad_output_min = grad_output.narrow(2, R, R)

            grad_input = grad_output.new().resize_as_(input).fill_(0)
    
            grad_input.scatter_(2, ctx.indices_max, grad_output_max)
            grad_input.scatter_(2, ctx.indices_min, grad_output_min)
    
            return grad_input, None
    
class MinMax(nn.Module):
    def __init__(self, R):
        super(MinMax, self).__init__()
        self.R = R

    def forward(self, input):
        return MinMaxFunction.apply(input, self.R)

    def __repr__(self):
        return self.__class__.__name__ + f'(R={self.R}))'


## TO DO
#
#

if __name__ == '__main__':
    

    model = GetModel(2048, J=1, R=2, n_first_mlp_neurons=200, n_second_mlp_neurons=100)
    print(model.min_max_layer.weights)

    # from dataset import DataFeatures
    # root = "/Users/rezachi/ML/datasets/owkin/data/"
    # data = DataFeatures(root)
    # X_train, y_train, centers_train, patients_train = data.__getdata__()

    # x = model(X_train)
    # print(x.shape)

    # print('\n    Total params: %.2f No' % (sum(p.numel() for p in model.parameters())))
    # print('    Total trainable params: %.0f No' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))
