"""Module for training a reward model from the generated feedback."""

import math
import pickle
from os import path
from pathlib import Path
from random import randint, randrange
from typing import Union

import numpy
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

import wandb

from .common import (
    ALGORITHM,
    ENSEMBLE_COUNT,
    ENVIRONMENT_NAME,
    FEEDBACK_ID,
    FEEDBACK_TYPE,
    STEPS_PER_CHECKPOINT,
    USE_REWARD_DIFFERENCE,
    USE_SDE,
    cpu_count,
    get_reward_model_name,
)
from .networks import LightningNetwork, calculate_mle_loss, calculate_mse_loss
from .types import Feedback, FeedbackType

REWARD_MODEL_ID = get_reward_model_name(f"{randrange(1000, 10000)}")

script_path = Path(__file__).parent.resolve()

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(self, dataset_path: str, feedback_type: FeedbackType):
        """Initialize dataset."""
        print("Loading dataset...")

        with open(dataset_path, "rb") as feedback_file:
            feedback_list: list[Feedback] = pickle.load(feedback_file)

        expert_value_key = (
            "expert_value" if not USE_REWARD_DIFFERENCE else "expert_value_difference"
        )

        match feedback_type:
            case "evaluative":
                # First: Observation, Second: Reward
                self.first = [
                    numpy.concatenate(
                        [feedback["observation"], feedback["actions"]]
                    ).astype("float32")
                    for feedback in feedback_list
                ]
                self.second = [
                    numpy.float32(feedback[expert_value_key])
                    for feedback in feedback_list
                ]
            case "comparative":
                # First: high-reward observation, Second: low-reward observation
                observation_pairs = [
                    map(
                        lambda feedback: numpy.concatenate(
                            [feedback["observation"], feedback["actions"]]
                        ).astype("float32"),
                        sorted(
                            list(
                                (
                                    feedback_list[randint(0, len(feedback_list) - 1)],
                                    feedback_list[randint(0, len(feedback_list) - 1)],
                                )
                            ),
                            key=lambda feedback: feedback[expert_value_key],
                            reverse=True,
                        ),
                    )
                    for _ in range(len(feedback_list))
                ]

                self.first, self.second = zip(*observation_pairs)
            case "corrective" | "demonstrative":
                demonstrative_length = (
                    STEPS_PER_CHECKPOINT if feedback_type == "demonstrative" else None
                )

                # First: Expert's observation, Second: Agent's observation
                # Note: this only works for IRL (for observation-only models, use
                # `expert_observation` and `next_observation` for the first and second respectively)
                # TODO: experiment with changing the threshold
                self.first = [
                    numpy.concatenate(
                        [feedback["expert_observation"], feedback["expert_actions"]]
                    ).astype("float32")
                    for feedback in feedback_list
                    if feedback["expert_own_value"] > feedback["expert_value"]
                ][:demonstrative_length]

                self.second = [
                    numpy.concatenate(
                        [feedback["observation"], feedback["actions"]]
                    ).astype("float32")
                    for feedback in feedback_list
                    if feedback["expert_own_value"] > feedback["expert_value"]
                ][:demonstrative_length]
            case "descriptive":
                # First: Changed observation, Second: Agent's observation
                # TODO: generate more perturbation for one feedback
                model_inputs = numpy.array(
                    list(
                        map(
                            lambda feedback: numpy.concatenate(
                                [feedback["observation"], feedback["actions"]]
                            ).astype("float32"),
                            feedback_list,
                        )
                    )
                )
                standard_deviation = model_inputs.std(axis=0)

                for index, feedback in enumerate(feedback_list):
                    perturbations = numpy.random.normal(
                        0, standard_deviation, model_inputs.shape[-1]
                    )

                    perturbations[feedback["expert_value_attributions"] > 0] = 0

                    model_inputs[index] += perturbations

                # First: Observation, Second: Reward
                self.first = model_inputs
                self.second = [
                    numpy.float32(feedback[expert_value_key])
                    for feedback in feedback_list
                ]
            case _:
                raise NotImplementedError(
                    "Dataset not implemented for this feedback type ."
                )

        print("Dataset loaded")

    def __len__(self):
        """Return size of dataset."""
        return len(self.first)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.first[index], self.second[index]


def train_reward_model(
    reward_model: LightningModule,
    dataset: FeedbackDataset,
    maximum_epochs: int,
    batch_size: int,
    gradient_clip_value: Union[float, None] = None,
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
        # Ensemble needs to have a batch size divisible by the ensemble count
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cpu_count,
        # Ensemble needs to have a batch size divisible by the ensemble count
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(script_path, "..", "reward_model_checkpoints"),
        filename=REWARD_MODEL_ID,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="Masters Thesis", name=REWARD_MODEL_ID)

    # add your batch size to the wandb config
    wandb_logger.experiment.config.update(
        {
            "rl_algorithm": ALGORITHM,
            "rl_environment": ENVIRONMENT_NAME,
            "rl_is_use_sde": USE_SDE,
            "rl_feedback_type": FEEDBACK_TYPE,
            "max_epochs": maximum_epochs,
            "batch_size": batch_size,
            "gradient_clip_value": gradient_clip_value,
            "learning_rate": reward_model.learning_rate,
        }
    )

    trainer = Trainer(
        max_epochs=maximum_epochs,
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            checkpoint_callback,
            *([callback] if callback is not None else []),
        ],
    )

    trainer.fit(reward_model, train_loader, val_loader)

    wandb.finish()

    return reward_model


def main():
    """Run reward model training."""
    print("Feedback ID:", FEEDBACK_ID)
    print("Model ID:", REWARD_MODEL_ID)
    print()

    # Load data
    dataset = FeedbackDataset(
        path.join(script_path, "..", "feedback", f"{FEEDBACK_ID}.pkl"), FEEDBACK_TYPE
    )

    # Select loss function based on feedback type
    loss_function = None

    match FEEDBACK_TYPE:
        case "evaluative" | "descriptive":
            loss_function = calculate_mse_loss
        case "comparative" | "corrective" | "demonstrative":
            loss_function = calculate_mle_loss
        case _:
            raise NotImplementedError(
                "Loss function not implemented for this feedback type."
            )

    # Train reward model
    reward_model = LightningNetwork(
        input_dim=23,
        hidden_dim=256,
        layer_num=12,
        output_dim=1,
        loss_function=loss_function,
        learning_rate=(
            1e-6
            if FEEDBACK_TYPE == "corrective"
            else (5e-6 if FEEDBACK_TYPE == "comparative" else 2e-5)
        ),
        ensemble_count=ENSEMBLE_COUNT,
    )

    train_reward_model(reward_model, dataset, maximum_epochs=100, batch_size=4)


if __name__ == "__main__":
    main()
