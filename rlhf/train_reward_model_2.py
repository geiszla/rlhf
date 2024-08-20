"""Module for training a reward model from the generated feedback."""

import argparse
import math
import os
import pickle
from os import path
from pathlib import Path
from random import randint, randrange
from typing import Union, List, Tuple
from numpy.typing import NDArray
import gymnasium as gym
import random

import numpy
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

import wandb
from rlhf.common import get_reward_model_name
from rlhf.datatypes import FeedbackDataset, FeedbackType, SegmentT
from rlhf.networks import LightningNetwork, LightingCnnNetwork, calculate_pairwise_loss, calculate_single_reward_loss

script_path = Path(__file__).parents[1].resolve()

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


class FeedbackDataset(Dataset):
    """PyTorch Dataset for loading the feedback data."""

    def __init__(
        self,
        dataset_path: str,
        feedback_type: FeedbackType,
    ):
        """Initialize dataset."""
        print("Loading dataset...")

        self.targets: Union[List[SegmentT], List[NDArray], Tuple[SegmentT, SegmentT], Tuple[NDArray, NDArray]] = []
        self.preds: List[int] = []

        with open(dataset_path, "rb") as feedback_file:
            feedback_data: FeedbackDataset = pickle.load(feedback_file)

        match feedback_type:
            case "evaluative":
                for seg in feedback_data["segments"]:
                    obs = torch.vstack([torch.as_tensor(p[0]).float() for o in seg])
                    actions = torch.vstack([torch.as_tensor(p[1]).float() for o in seg])
                    self.targets.append((obs, actions))
                    self.preds = feedback_data["ratings"]
            case "comparative":
                for comp in feedback_data["preferences"]:
                    if comp[2] < 1:
                        self.targets.append((feedback_data["segments"][comp[1]], feedback_data["segments"][comp[0]]))
                    else:
                        self.targets.append((feedback_data["segments"][comp[0]], feedback_data["segments"][comp[1]]))
                    self.preds.append(abs(comp[2]))
                    
            case "demonstrative":
                for demo in demos:
                    obs = torch.vstack([torch.as_tensor(p[0]).float() for p in demo])
                    actions = torch.vstack([torch.as_tensor(p[1]).float() for p in demo])

                    # just use a random segment as the opposite
                    rand_index = random.randrange(0, len(segments))
                    obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][rand_index]])
                    actions2 = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][rand_index]])
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(1) # assume that the demonstration is optimal, maybe add confidence value (based on regret)
            case "corrective":
                for comp in feedback_data["corrections"]:
                    comp_index_gt = comp[0]
                    obs = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][comp_index_gt]])
                    actions = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][comp_index_gt]])

                    # just use a random segment as the opposite
                    comp_index_corr = comp[1]
                    obs2 = torch.vstack([torch.as_tensor(p[0]).float() for p in feedback_data["segments"][comp_index_corr]])
                    actions2 = torch.vstack([torch.as_tensor(p[1]).float() for p in feedback_data["segments"][comp_index_corr]])
                    
                    self.targets.append(((obs, actions),(obs2, actions2)))
                    self.preds.append(1) # because the second element is the correction    
            case "descriptive":
                for desc in feedback_data["description"]:
                    self.targets.append(torch.as_tensor(desc[0]).float())
                    self.preds.append(desc[1])
            case "description_preference":
                for dpref in feedback_data["description_preference"]:  
                    if dpref[2] < 1:
                        self.targets.append((feedback_data["descriptions"][dpref[0]][1],  feedback_data["descriptions"][dpref[0]][0]))
                    else:
                        self.targets.append((feedback_data["descriptions"][dpref[0]][0],  feedback_data["descriptions"][dpref[0]][1]))
                    self.preds.append(abs(dpref[1]))
            case _:
                raise NotImplementedError(
                    "Dataset not implemented for this feedback type."
                )

        print("Dataset loaded")

    def __len__(self):
        """Return size of dataset."""
        return len(self.targets)

    def __getitem__(self, index):
        """Return item with given index."""
        return self.targets[index], self.preds[index]


def train_reward_model(
    reward_model: LightningModule,
    reward_model_id: str,
    feedback_type: FeedbackType,
    dataset: FeedbackDataset,
    maximum_epochs: int = 100,
    batch_size: int = 1,
    cpu_count: int = 4,
    algorithm: str = "sac",
    environment: str = "HalfCheetah-v3",
    gradient_clip_value: Union[float, None] = None,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
):

    get_reward_model_name(
        reward_model_id,
        feedback_type,
        f"{randrange(1000, 10000)}",
    )

    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[training_set_size, len(dataset) - training_set_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count,
    )

    val_loader = DataLoader(
        val_set, batch_size=1, pin_memory=True, num_workers=cpu_count
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=path.join(script_path, "..", "reward_model_checkpoints"),
        filename=reward_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="multi_reward_feedback", name=reward_model_id)

    trainer = Trainer(
        max_epochs=maximum_epochs,
        devices=[0],
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
        accumulate_grad_batches=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            checkpoint_callback,
            *([callback] if callback is not None else []),
        ],
    )

    # add your batch size to the wandb config
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(
            {
                "rl_algorithm": algorithm,
                "rl_environment": environment,
                "rl_feedback_type": feedback_type,
                "max_epochs": maximum_epochs,
                #"batch_size": batch_size,
                "gradient_clip_value": gradient_clip_value,
                "learning_rate": reward_model.learning_rate,
            }
        )

    trainer.fit(reward_model, train_loader, val_loader)

    wandb.finish()

    return reward_model


def main():

    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--experiment", type=int, default=0, help="Experiment number"
    )
    arg_parser.add_argument(
        "--feedback_type",
        type=str,
        default="evaluative",
        help="Type of feedback to train the reward model",
    )
    arg_parser.add_argument(
        "--algorithm",
        type=str,
        default="sac",
        help="RL algorithm used to generate the feedback",
    )
    arg_parser.add_argument(
        "--environment",
        type=str,
        default="HalfCheetah-v3",
        help="Environment used to generate the feedback",
    )
    arg_parser.add_argument(
        "--use-sde",
        type=bool,
        default=False,
        help="Whether the RL algorithm used SDE",
    )
    arg_parser.add_argument(
        "--use-reward-difference",
        type=bool,
        default=False,
        help="Whether to use the reward difference",
    )
    arg_parser.add_argument(
        "--steps-per-checkpoint",
        type=int,
        default=10000,
        help="Number of steps per checkpoint",
    )
    args = arg_parser.parse_args()

    FEEDBACK_ID = "_".join(
        [args.algorithm, args.environment]
    )
    MODEL_ID = f"#{args.experiment}_{FEEDBACK_ID}"

    # Load data
    dataset = FeedbackDataset(
        path.join(script_path, "feedback", f"{FEEDBACK_ID}.pkl"),
        args.feedback_type
    )

    # Select loss function based on feedback type
    loss_function = None
    architecture_cls = None

    match args.feedback_type:
        case "evaluative" | "descriptive":
            loss_function = calculate_single_reward_loss
        case "comparative" | "corrective" | "demonstrative" | "demonstrative_preference":
            loss_function = calculate_pairwise_loss
        case _:
            raise NotImplementedError(
                "Loss function not implemented for this feedback type."
            )

    match args.environment:
        case "Ant-v5" | "Humanoid-v5" | "HalfCheetah-v5":
            architecture_cls = LightningNetwork
            
        case "MiniGrid-GoToDoor-5x5-v0" | "procgen-coinrun-v0" | "procgen-miner-v0" | "procgen-maze-v0" | "ALE/MsPacman-v5":
            architecture_cls = LightingCnnNetwork
    
    env = gym.make(args.environment)
    
    # Train reward model
    reward_model = architecture_cls(
        input_spaces=(env.observation_space, env.action_space),
        hidden_dim=256,
        action_hidden_dim=32,
        layer_num=6,
        output_dim=1,
        loss_function=loss_function,
        learning_rate=(
            1e-5
            #1e-6
            #if args.feedback_type == "corrective"
            #else (1e-5 if args.feedback_type == "comparative" else 2e-5)
        ),
    )

    train_reward_model(
        reward_model,
        MODEL_ID,
        args.feedback_type,
        dataset,
        maximum_epochs=100,
        batch_size=4,
        split_ratio=0.5,
        cpu_count=cpu_count,
    )


if __name__ == "__main__":
    main()
