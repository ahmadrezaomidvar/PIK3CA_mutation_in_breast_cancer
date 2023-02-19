"""Chowder model."""

import torch.nn as nn
import torch
from typing import List, Optional


class Chowder(nn.Module):
    """Chowder model.
    Courtiol P, Tramel EW, Sanselme M, Wainrib G.
    Classification and Disease Localization in Histopathology Using Only Global Labels:
    A Weakly-Supervised Approach. 2018 [cited 2023 Feb 15];Available from: https://arxiv.org/abs/1802.02212v2
    """

    def __init__(
        self,
        features_dim: int,
        n_kernels: int,
        retained_features: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        n_first_mlp_neurons: int = 200,
        n_second_mlp_neurons: int = 100,
        reduce_method="minmax",
    ) -> None:
        """
        Initialize Chowder model.

        Args:
          features_dim: the number of features in the input data
          n_kernels: the number of embedding channels (kernels)
          retained_features: the number of min/max retained values
          n_first_mlp_neurons: number of neurons in the first layer of the MLP. Defaults to 200
          n_second_mlp_neurons: number of neurons in the second layer of the MLP. Defaults to 100
        """
        super().__init__()

        self.embedding_layer = nn.Conv1d(features_dim, n_kernels, 1)

        if reduce_method == "minmax":
            if not retained_features:
                raise ValueError(
                    "retained_features needs to be specified for reduced method minmax."
                )
            self.reduce_layer = MinMaxModule(retained_features=retained_features)
            next_dim = 2 * retained_features

        elif reduce_method == "quantile":
            if not quantiles:
                raise ValueError(
                    "quantiles needs to be specified for reduced method quantile."
                )
            self.reduce_layer = QuantilesModule(quantiles=quantiles)
            next_dim = len(quantiles)

        else:
            raise ValueError(f"Unknown reduce method: {reduce_method}")

        self.mlp = nn.Sequential(
            nn.Linear(next_dim * n_kernels, n_first_mlp_neurons),
            nn.Sigmoid(),
            nn.Linear(n_first_mlp_neurons, n_second_mlp_neurons),
            nn.Sigmoid(),
            nn.Linear(n_second_mlp_neurons, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
          x : the input data

        Returns:
          The logits of the model.
        """

        x = self.embedding_layer(x)
        x = self.reduce_layer(x)
        x = x.view(x.size(0), -1)
        logits = self.mlp(x)

        return logits


class MinMaxModule(nn.Module):
    """MinMaxModule."""

    def __init__(self, retained_features: int) -> None:
        """
        Initialize MinMaxModule.

        Args:
          retained_features: the number of min/max retained values.
        """
        super().__init__()
        self.retained_features = retained_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The forward method to take the top k values from the input tensor, and the bottom k values from the input tensor,
        and concatenate them together

        Args:
          input: the input tensor

        Returns:
          The output is a tensor of the same shape as the input except last dim, but with the top k and bottom k values
        sorted in descending order.
        """

        top_max_k, _ = torch.topk(input, dim=2, k=self.retained_features)

        input_negative = torch.negative(input)
        top_min_k_neg, _ = torch.topk(input_negative, dim=2, k=self.retained_features)
        top_min_k = torch.negative(top_min_k_neg)

        output = torch.cat((top_max_k, top_min_k), dim=2)
        output, _ = torch.sort(output, dim=2, descending=True)

        return output


# This is an improvement idea
# (not implemented in the original paper)
class QuantilesModule(nn.Module):
    """QuantilesModule."""

    def __init__(self, quantiles: List[float]) -> None:
        """Construct QuantilesModule"""
        super().__init__()
        self.quantiles = quantiles

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward implementation"""
        output = torch.quantile(input=input, q=torch.tensor(self.quantiles), dim=2)
        output = torch.swapaxes(output, 0, 1)
        output = torch.swapaxes(output, 1, 2)
        return output


if __name__ == "__main__":
    import numpy as np

    # Test the model
    input = torch.rand(30, 2048, 1000)
    model = Chowder(
        features_dim=2048, n_kernels=10, retained_features=10, reduce_method="minmax"
    )
    output = model(input)
    print(output.shape)

    # Test the model with quantiles
    input = torch.rand(30, 2048, 1000)
    model = Chowder(
        features_dim=2048, n_kernels=10, quantiles=[0.1, 0.9], reduce_method="quantile"
    )
    output = model(input)
    print(output.shape)
