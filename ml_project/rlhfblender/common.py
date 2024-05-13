"""Utility function for the RLHF Blender project."""

import os
from os import path
from pathlib import Path
from typing import Literal, Union

import torch

from ..types import FeedbackType

ALGORITHM: Union[Literal["sac"], Literal["ppo"]] = "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_SDE = True
USE_REWARD_MODEL = False

STEPS_PER_CHECKPOINT = 10000

MODEL_ID = "_".join(
    [
        ALGORITHM,
        ENVIRONMENT_NAME,
        *(["sde"] if USE_SDE else []),
        *(["finetuned"] if USE_REWARD_MODEL else []),
    ]
)

FEEDBACK_TYPE: FeedbackType = "evaluative"

script_path = Path(__file__).parent.resolve()
checkpoints_path = path.join(script_path, "..", "rl", "models_final")

cpu_count = os.cpu_count()
cpu_count = cpu_count if cpu_count is not None else 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_reward_model_name(number_postfix: int):
    """Return the name of the trained reward model by the number postfix."""
    return "_".join([MODEL_ID, FEEDBACK_TYPE, str(number_postfix)])
