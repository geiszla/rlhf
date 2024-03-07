"""Utility function for the RLHF Blender project."""

from os import path
from pathlib import Path

script_path = Path(__file__).parent.resolve()
checkpoints_path = path.join(script_path, "..", "rl", "models_final")

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
