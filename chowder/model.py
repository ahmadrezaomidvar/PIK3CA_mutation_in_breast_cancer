import torch.nn as nn
import torch


class Chowder(nn.Module):
    """Chowder model."""

    def __init__(
        self,
        features_dim: int,
        J: int,
        R: int,
        n_first_mlp_neurons: int = 200,
        n_second_mlp_neurons: int = 100,
    ) -> None:
        """
        Initialize Chowder model.

        Args:
          features_dim (int): the number of features in the input data
          J (int): the number of embedding channels
          R (int): the number of min/max retained values
          n_first_mlp_neurons (int): number of neurons in the first layer of the MLP. Defaults to 200
          n_second_mlp_neurons (int): number of neurons in the second layer of the MLP. Defaults to 100
        """
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
        """
        Forward pass of the model.

        Args:
          x (torch.Tensor): the input data

        Returns:
          The logits of the model.
        """

        x = self.embedding_layer(x)
        x = self.min_max_layer(x)
        x = x.view(x.size(0), -1)
        logits = self.mlp(x)

        return logits


class MinMaxModule(nn.Module):
    """MinMaxModule."""

    def __init__(self, k: int) -> None:
        """
        Initialize MinMaxModule.

        Args:
          k (int): the number of min/max retained values.
        """
        super().__init__()
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The forward method to take the top k values from the input tensor, and the bottom k values from the input tensor,
        and concatenate them together

        Args:
          input (torch.Tensor): the input tensor

        Returns:
          The output is a tensor of the same shape as the input except last dim, but with the top k and bottom k values
        sorted in descending order.
        """

        top_max_k, _ = torch.topk(input, dim=2, k=self.k)

        input_negative = torch.negative(input)
        top_min_k_neg, _ = torch.topk(input_negative, dim=2, k=self.k)
        top_min_k = torch.negative(top_min_k_neg)

        output = torch.cat((top_max_k, top_min_k), dim=2)
        output, _ = torch.sort(output, dim=2, descending=True)

        return output
