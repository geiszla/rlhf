"""Module for fine-tuning a reward model using preference data."""

import math
import os
from os import path
from pathlib import Path
from typing import Literal, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split

from .datasets import MultiStepPreferenceDataset, PreferenceDataset
from .networks import LightningTrajectoryNetwork

# from .networks_old import LightningRNNNetwork

ALGORITHM = "sac"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_REWARD_MODEL = False
USE_SDE = True

model_id = f"{ALGORITHM}_{ENVIRONMENT_NAME}"
model_id += "_sde" if USE_SDE else ""
model_id += "_finetuned" if USE_REWARD_MODEL else ""

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")

# File paths
script_path = Path(__file__).parent.resolve()
file_path = path.join(script_path, "preference_dataset.pkl")
models_path = path.join(script_path, "models_final")
pretrained_model_path = path.join(models_path, f"{model_id}_pretrained.ckpt")

cpu_count = os.cpu_count()
cpu_count = cpu_count if cpu_count is not None else 8


def train_reward_model(
    reward_model: LightningModule,
    name: Union[
        Literal["mlp_single"], Literal["mlp"], Literal["mlp_finetuned"], Literal["rnn"]
    ],
    dataset: Union[PreferenceDataset, MultiStepPreferenceDataset],
    epochs: int,
    batch_size: int,
    split_ratio: float = 0.8,
    callback: Union[Callback, None] = None,
    enable_progress_bar=True,
):
    """Train a reward model given preference data."""
    train_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[train_size, len(dataset) - train_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cpu_count,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=models_path,
        filename=f"{model_id}_{name}",
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
    """Run reward model fine-tuning."""
    # Train MLP using single observations
    # dataset = PreferenceDataset(file_path)

    # reward_model = LightningNetwork(
    #     input_dim=17, hidden_dim=256, layer_num=3, output_dim=1
    # )

    # train_reward_model(reward_model, "mlp_single", dataset, epochs=100, batch_size=4)

    # Finetune MLP using full-trajectory loss
    dataset = MultiStepPreferenceDataset(file_path, sequence_length=70)

    reward_model = LightningTrajectoryNetwork.load_from_checkpoint(
        pretrained_model_path, input_dim=17, hidden_dim=256, layer_num=12, output_dim=1
    )

    train_reward_model(reward_model, "mlp_finetuned", dataset, epochs=100, batch_size=4)

    # Train MLP using full-trajectory loss
    # dataset = MultiStepPreferenceDataset(file_path, sequence_length=70)

    # reward_model = LightningTrajectoryNetwork(
    #     input_dim=17, hidden_dim=256, layer_num=12, output_dim=1
    # )

    # train_reward_model(reward_model, "mlp", dataset, epochs=100, batch_size=4)

    # Train RNN
    # dataset = MultiStepPreferenceDataset(file_path, sequence_length=70)

    # reward_model = LightningRNNNetwork(
    #     input_dim=17, hidden_dim=256, layer_num=12, output_dim=1, dropout=0.2
    # )

    # train_reward_model(reward_model, "rnn", dataset, epochs=100, batch_size=4)


if __name__ == "__main__":
    main()
