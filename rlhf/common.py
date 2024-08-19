"""Utility function for the RLHF Blender project."""

import os
from os import path
from pathlib import Path
from typing import Literal, Union

import torch

from .types import FeedbackType

# Set these two before each experiment
EXPERIMENT_NUBMER = 10
FEEDBACK_TYPE: FeedbackType = "evaluative"

# Additional configuration options
ALGORITHM: Union[Literal["sac"], Literal["ppo"]] = "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_SDE = False
USE_REWARD_DIFFERENCE = False
ENSEMBLE_COUNT = 4

STEPS_PER_CHECKPOINT = 10000

# Paths and other variables used by multiple scripts
FEEDBACK_ID = "_".join([ALGORITHM, ENVIRONMENT_NAME, *(["sde"] if USE_SDE else [])])
MODEL_ID = f"#{EXPERIMENT_NUBMER}_{FEEDBACK_ID}"

script_path = Path(__file__).parent.resolve()
checkpoints_path = path.join(script_path, "..", "rl_checkpoints")

cpu_count = os.cpu_count()
cpu_count = cpu_count if cpu_count is not None else 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Common functions
def get_reward_model_name(
    postfix: int | str,
    feedback_override: Union[FeedbackType, Literal["without"]] | None = None,
):
    """Return the name of the trained reward model by the number postfix."""
    return "_".join(
        [
            MODEL_ID,
            *(
                [feedback_override or FEEDBACK_TYPE]
                if feedback_override != "without"
                else []
            ),
            *(["ens"] if ENSEMBLE_COUNT > 1 else []),
            *(
                ["diff"]
                if USE_REWARD_DIFFERENCE and feedback_override != "without"
                else []
            ),
            str(postfix),
        ]
    )
