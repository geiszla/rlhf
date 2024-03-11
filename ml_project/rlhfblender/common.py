"""Utility function for the RLHF Blender project."""

import os
from os import path
from pathlib import Path

ALGORITHM = "sac"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v3"
USE_REWARD_MODEL = False
USE_SDE = True

STEPS_PER_CHECKPOINT = 10000

MODEL_ID = "_".join(
    [
        ALGORITHM,
        ENVIRONMENT_NAME,
        *(["sde"] if USE_SDE else []),
        *(["finetuned"] if USE_REWARD_MODEL else []),
    ]
)

script_path = Path(__file__).parent.resolve()
checkpoints_path = path.join(script_path, "..", "rl", "models_final")

cpu_count = os.cpu_count()
cpu_count = cpu_count if cpu_count is not None else 8
