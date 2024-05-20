"""Module for training an RL agent."""

from os import path
from typing import Union

import numpy as np
import torch
from imitation.rewards.reward_function import RewardFn
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ml_project.reward_model.networks_old import LightningNetwork

from ..types import FeedbackType
from .common import (
    ALGORITHM,
    DEVICE,
    ENVIRONMENT_NAME,
    FEEDBACK_TYPE,
    MODEL_ID,
    USE_SDE,
    checkpoints_path,
    cpu_count,
    script_path,
)

# Set feedback type to None to not use the custom reward model
TRAINING_FEEDBACK_TYPE: Union[FeedbackType, None] = FEEDBACK_TYPE

tensorboard_path = path.join(script_path, "..", "..", "rl_logs")

REWARD_MODEL_ID = "_".join(
    [MODEL_ID, *([FEEDBACK_TYPE] if FEEDBACK_TYPE is not None else []), str(3653)]
)

reward_model_path = path.join(
    script_path, "reward_model_checkpoints", f"{REWARD_MODEL_ID}.ckpt"
)


class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(self):
        """Initialize custom reward."""
        super().__init__()

        # pylint: disable=no-value-for-parameter
        self.reward_model = LightningNetwork.load_from_checkpoint(
            checkpoint_path=reward_model_path
        )

    def __call__(
        self,
        state: np.ndarray,
        _actions: np.ndarray,
        _next_state: np.ndarray,
        _done: np.ndarray,
    ) -> list:
        """Return reward given the current state."""
        rewards = self.reward_model(torch.Tensor(state).to(DEVICE))

        return [reward.detach().item() for reward in rewards]


def main():
    """Run RL agent training."""
    # For PPO, the more environments there are, the more `num_timesteps` shifts
    # from `total_timesteps`
    environment = make_vec_env(
        ENVIRONMENT_NAME, n_envs=cpu_count if ALGORITHM != "ppo" else 1
    )

    if FEEDBACK_TYPE is not None:
        environment = RewardVecEnvWrapper(environment, reward_fn=CustomReward())

    # Select agent algorithm
    if ALGORITHM == "sac":
        model_class = SAC
    elif ALGORITHM == "ppo":
        model_class = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    model = model_class(
        "MlpPolicy",
        environment,
        verbose=1,
        tensorboard_log=tensorboard_path,
        use_sde=USE_SDE,
    )

    iterations = 20
    steps_per_iteration = 25000
    timesteps = 0

    for iteration_count in range(iterations):
        trained_model = model.learn(
            total_timesteps=steps_per_iteration * (iteration_count + 1) - timesteps,
            reset_num_timesteps=False,
            tb_log_name=REWARD_MODEL_ID,
        )

        timesteps = trained_model.num_timesteps

        model.save(path.join(checkpoints_path, f"{REWARD_MODEL_ID}_{timesteps}"))


if __name__ == "__main__":
    main()
