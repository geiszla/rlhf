"""Module for instantiating a neural network."""

# pylint: disable=arguments-differ
from typing import Type, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn


def calculate_multi_reward_loss(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    rewards1 = torch.sum(network(batch[0]))
    rewards2 = torch.sum(network(batch[1]))

    index_of_preferred_traj = 1
    softmax = torch.softmax(torch.cat((rewards1, rewards2), 1), 1)[
        :, index_of_preferred_traj
    ]
    loss = -torch.sum(torch.log(softmax))

    return loss


def calculate_pretrain_loss(network: LightningModule, batch: Tensor):
    """Calculate the mean squared error between reward and predicted reward."""
    loss_fn = nn.MSELoss(reduce=sum)
    obs, reward = batch
    obs = obs.unsqueeze(
        dim=1
    )  # maybe remove if used for something else than experiment
    loss = loss_fn(
        network(obs.float()),
        reward.float().unsqueeze(1),
    )
    return loss


def calculate_multi_reward_loss_experiment(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    rewards1 = network(batch[0]).unsqueeze(dim=1)
    rewards2 = network(batch[1]).unsqueeze(dim=1)

    index_of_preferred_traj = 1
    softmax = torch.softmax(torch.cat((rewards1, rewards2), 1), 1)[
        :, index_of_preferred_traj
    ]
    loss = -torch.sum(torch.log(softmax + 1e-6))

    return loss


class LightningTrajectoryNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        layer_num: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        learning_rate=1e-4,
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
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

    def forward(self, batch: Tensor):
        """Do a forward pass through the neural network (inference)."""
        return [
            self.network(state_batch) for state_batch in torch.swapaxes(batch, 0, 1)
        ]

    def training_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for training."""
        loss = calculate_multi_reward_loss(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = calculate_multi_reward_loss(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        return optimizer


class LightningTrajectoryNetworkExperiment(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        layer_num: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
    ):
        super().__init__()

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
                layer.bias.data.zero_()

    def forward(self, batch: Tensor):
        """Do a forward pass through the neural network (inference)."""
        return sum(
            self.network(state_batch) for state_batch in torch.swapaxes(batch, 0, 1)
        )

    def training_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for training."""
        loss = calculate_multi_reward_loss_experiment(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = calculate_multi_reward_loss_experiment(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-3)
        return optimizer


class LightningTrajectoryNetworkPreTrain(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        layer_num: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
    ):
        super().__init__()

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
                layer.bias.data.zero_()

    def forward(self, batch: Tensor):
        """Do a forward pass through the neural network (inference)."""
        return self.network(batch)

    def training_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for training."""
        loss = calculate_pretrain_loss(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = calculate_pretrain_loss(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
