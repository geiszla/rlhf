"""Module for instantiating a neural network."""

# pylint: disable=arguments-differ
from typing import Callable, Type, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.functional import mse_loss


def calculate_mse_loss(network: LightningModule, batch: Tensor):
    """Calculate the mean squared erro loss for the reward."""
    return mse_loss(network(batch[0]), batch[1].unsqueeze(1), reduction="sum")


def calculate_mle_loss(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    rewards1 = network(batch[0]).flatten()
    rewards2 = network(batch[1]).flatten()

    probs_softmax = torch.exp(rewards1) / (torch.exp(rewards1) + torch.exp(rewards2))

    loss = -torch.sum(torch.log(probs_softmax))

    return loss


class LightningRNNNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_num: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Initialize the network
        self.lstm = nn.LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layer_num=layer_num,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_dim, output_dim)

        # Initialize the weights
        # https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, parameter in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter.data)
                elif "bias_ih" in name:
                    parameter.data.fill_(0)
                    # Set forget-gate bias to 1
                    forget_size = parameter.size(0)
                    parameter.data[(forget_size // 4) : (forget_size // 2)].fill_(1)
                elif "bias_hh" in name:
                    parameter.data.fill_(0)
            elif "linear" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(parameter.data)
                elif "bias" in name:
                    parameter.data.fill_(0)

    def forward(self, batch: Tensor):
        """Do a forward pass through the neural network (inference)."""
        lstm_out, _ = self.lstm(batch)
        prediction = self.linear(lstm_out[:, -1])
        return prediction

    def training_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for training."""
        loss = calculate_mle_loss(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = calculate_mle_loss(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


class LightningNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        layer_num: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
    ):
        super().__init__()

        self.loss_function = loss_function

        # Initialize the network
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)

        layers: list[nn.Module] = []

        for idx in range(len(layers_unit) - 1):
            layers.append(nn.Linear(layers_unit[idx], layers_unit[idx + 1]))
            layers.append(activation_function())

        layers.append(nn.Linear(layers_unit[-1], output_dim))

        if last_activation is not None:
            layers.append(last_activation())

        self.network = nn.Sequential(*layers)

        # Initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

                layer.bias.data.zero_()

        self.save_hyperparameters()

    def forward(self, batch: Tensor):
        """Do a forward pass through the neural network (inference)."""
        batch = self.network(batch)
        return batch

    def training_step(self, batch: Tensor):
        """Compute the loss for training."""
        loss = self.loss_function(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class Network(nn.Module):
    """Neural network to model the RL agent's reward."""

    def __init__(
        self,
        layer_num,
        input_dim,
        output_dim,
        hidden_dim,
        activation_function=nn.ReLU,
        last_activation=None,
        learning_rate=2e-4,
    ):
        super().__init__()

        self.learning_rate = learning_rate

        # Initialize the network
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)

        layers: list[nn.Module] = []

        for idx in range(len(layers_unit) - 1):
            layers.append(nn.Linear(layers_unit[idx], layers_unit[idx + 1]))
            layers.append(activation_function())

        layers.append(nn.Linear(layers_unit[-1], output_dim))

        if last_activation is not None:
            layers.append(last_activation())

        self.network = nn.Sequential(*layers)

        # Initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

                layer.bias.data.zero_()

    def forward(self, data: Tensor):
        """Do a forward pass through the neural network (inference)."""
        return self._forward(data)

    def _forward(self, data: Tensor):
        return self.network(data)

    def network_init(self):
        """Initialize the neural network."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
