"""Module for training a reward model from the generated feedback."""

import math
import pickle
from os import path
from pathlib import Path
from typing import Union

import numpy
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split

from ..reward_model.networks_old import LightningNetwork
from ..types import Feedback
from .common import MODEL_ID, cpu_count

FEEDBACK_TYPE = "evaluative"

script_path = Path(__file__).parent.resolve()

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(self, dataset_path: str):
        """Initialize dataset."""
        with open(dataset_path, "rb") as feedback_file:
            feedback_list: list[Feedback] = pickle.load(feedback_file)

        self.data = [
            feedback["observations"].astype("float32") for feedback in feedback_list
        ]
        self.target = [
            numpy.float32(feedback["expert_value"]) for feedback in feedback_list
        ]

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.data[index], self.target[index]


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
    dataset = FeedbackDataset(path.join(script_path, "feedback", f"{MODEL_ID}.pkl"))

    reward_model = LightningNetwork(
        input_dim=17, hidden_dim=256, layer_num=12, output_dim=1
    )

    train_reward_model(reward_model, dataset, epochs=100, batch_size=4)


if __name__ == "__main__":
    main()
