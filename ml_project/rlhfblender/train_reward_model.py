"""Module for training a reward model from the generated feedback."""

import math
import pickle
from os import path
from pathlib import Path
from random import randint
from typing import Literal, Union

import numpy
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split

from ..reward_model.networks_old import (
    LightningNetwork,
    calculate_mse_loss,
    calculate_single_reward_loss,
)
from ..types import Feedback
from .common import MODEL_ID, cpu_count

FeedbackType = Union[
    Literal["evaluative"],
    Literal["comparative"],
    Literal["corrective"],
    Literal["demonstrative"],
    Literal["descriptive"],
]

FEEDBACK_TYPE: FeedbackType = "comparative"

script_path = Path(__file__).parent.resolve()

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(self, dataset_path: str, feedback_type: FeedbackType):
        """Initialize dataset."""
        with open(dataset_path, "rb") as feedback_file:
            feedback_list: list[Feedback] = pickle.load(feedback_file)

        match feedback_type:
            case "evaluative":
                # First: Observations, Second: Reward
                self.first = [
                    feedback["observations"].astype("float32")
                    for feedback in feedback_list
                ]
                self.second = [
                    numpy.float32(feedback["expert_value"])
                    for feedback in feedback_list
                ]
            case "comparative":
                # First: high-reward observations, Second: low-reward observations
                observation_pairs = [
                    map(
                        lambda feedback: feedback["observations"].astype("float32"),
                        sorted(
                            list(
                                (
                                    feedback_list[randint(0, len(feedback_list) - 1)],
                                    feedback_list[randint(0, len(feedback_list) - 1)],
                                )
                            ),
                            key=lambda feedback: feedback["expert_value"],
                            reverse=True,
                        ),
                    )
                    for _ in range(len(feedback_list))
                ]

                self.first, self.second = zip(*observation_pairs)
            case _:
                raise NotImplementedError("Dataset for feedback type not implemented.")

    def __len__(self):
        """Return size of dataset."""
        return len(self.first)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.first[index], self.second[index]


def train_reward_model(
    reward_model: LightningModule,
    dataset: FeedbackDataset,
    epochs: int,
    batch_size: int,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
):
    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[training_set_size, len(dataset) - training_set_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count,
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, pin_memory=True, num_workers=cpu_count
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(script_path, "reward_model_checkpoints"),
        filename="_".join([MODEL_ID, FEEDBACK_TYPE]),
        monitor="val_loss",
    )

    trainer = Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        enable_progress_bar=enable_progress_bar,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            checkpoint_callback,
            *([callback] if callback is not None else []),
        ],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    return reward_model


def main():
    """Run reward model pre-training."""

    # Load data
    dataset = FeedbackDataset(
        path.join(script_path, "feedback", f"{MODEL_ID}.pkl"), FEEDBACK_TYPE
    )

    # Select loss function based on feedback type
    loss_function = None

    match FEEDBACK_TYPE:
        case "evaluative":
            loss_function = calculate_mse_loss
        case "comparative":
            loss_function = calculate_single_reward_loss
        case _:
            raise NotImplementedError(
                "Loss function for feedback type not implemented."
            )

    # Train reward model
    reward_model = LightningNetwork(
        input_dim=17,
        hidden_dim=256,
        layer_num=12,
        output_dim=1,
        loss_function=loss_function,
    )

    train_reward_model(reward_model, dataset, epochs=100, batch_size=4)


if __name__ == "__main__":
    main()
