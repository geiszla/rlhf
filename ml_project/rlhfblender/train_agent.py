"""Module for training an RL agent."""

from os import path
from typing import Literal, Union

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
    USE_SDE,
    cpu_count,
    get_reward_model_name,
    script_path,
)

# Set feedback type to None to not use the custom reward model
TRAINING_FEEDBACK_TYPE: Union[FeedbackType, Literal["expert"], None] = FEEDBACK_TYPE

tensorboard_path = path.join(script_path, "..", "..", "rl_logs")

REWARD_MODEL_ID = get_reward_model_name(
    "",
    is_without_feedback=(
        TRAINING_FEEDBACK_TYPE is None or TRAINING_FEEDBACK_TYPE == "expert"
    ),
)

reward_model_path = path.join(
    script_path, "reward_model_checkpoints", f"{REWARD_MODEL_ID}.ckpt"
)

if TRAINING_FEEDBACK_TYPE == "expert":
    # PPO
    # expert_model = PPO.load(
    #     path.join(
    #         script_path,
    #         "..",
    #         "..",
    #         "logs",
    #         "ppo",
    #         "HalfCheetah-v3_1",
    #         "HalfCheetah-v3.zip",
    #     ),
    #     custom_objects={
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     },
    # )

    # SAC
    expert_model = SAC.load(
        path.join(
            script_path,
            "..",
            "..",
            "logs",
            "sac",
            "HalfCheetah-v3_1",
            "HalfCheetah-v3.zip",
        ),
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )


class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(self):
        """Initialize custom reward."""
        super().__init__()

        if TRAINING_FEEDBACK_TYPE != "expert":
            # pylint: disable=no-value-for-parameter
            self.reward_model = LightningNetwork.load_from_checkpoint(
                checkpoint_path=reward_model_path
            )

    def __call__(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        next_state: np.ndarray,
        _done: np.ndarray,
    ) -> list:
        """Return reward given the current state."""

        if TRAINING_FEEDBACK_TYPE == "expert":
            # PPO
            # rewards = expert_model.policy.value_net(
            #     expert_model.policy.mlp_extractor(
            #         expert_model.policy.extract_features(
            #             torch.from_numpy(next_state).to(DEVICE)
            #         )
            #     )[0]
            # )

            # SAC
            rewards = torch.min(
                torch.cat(
                    expert_model.policy.critic_target(
                        torch.from_numpy(state).to(DEVICE),
                        torch.from_numpy(actions).to(DEVICE),
                    ),
                    dim=1,
                ),
                dim=1,
            )[0]
        else:
            rewards = self.reward_model(
                torch.cat(
                    [
                        torch.Tensor(state).to(DEVICE),
                        torch.Tensor(actions).to(DEVICE),
                    ],
                    dim=1,
                )
            )

        return [reward.detach().item() for reward in rewards]


def main():
    """Run RL agent training."""
    # For PPO, the more environments there are, the more `num_timesteps` shifts
    # from `total_timesteps`
    environment = make_vec_env(
        ENVIRONMENT_NAME, n_envs=cpu_count if ALGORITHM != "ppo" else 1
    )

    if TRAINING_FEEDBACK_TYPE is not None:
        print(f'Using the "{TRAINING_FEEDBACK_TYPE}" reward model for training.')
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
        # verbose=1,
        tensorboard_log=tensorboard_path,
        use_sde=USE_SDE,
        gamma=0,
    )

    iterations = 20
    steps_per_iteration = 125000
    timesteps = 0

    for iteration_count in range(iterations):
        trained_model = model.learn(
            total_timesteps=steps_per_iteration * (iteration_count + 1) - timesteps,
            reset_num_timesteps=False,
            tb_log_name=REWARD_MODEL_ID,
        )

        timesteps = trained_model.num_timesteps
        print(f"{timesteps}/{steps_per_iteration * iterations} steps done")

        # model.save(path.join(checkpoints_path, f"{REWARD_MODEL_ID}_{timesteps}"))


if __name__ == "__main__":
    main()
