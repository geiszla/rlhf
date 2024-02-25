"""Module for training an RL agent."""

# ruff: noqa: E402
# pylint: disable=wrong-import-position
import os
from os import path
from pathlib import Path

import numpy as np
import torch
from imitation.rewards.reward_function import RewardFn
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ml_project.reward_model.networks_old import LightningNetwork

from ..reward_model.networks import LightningTrajectoryNetwork
from ..reward_model.networks_old import LightningRNNNetwork

ALGORITHM = "sac"  # "ppo" or "sac"
ENVIRONMENT_NAME = "HalfCheetah-v4"
REWARD_MODEL = None  # "mlp_single", "mlp", "mlp_finetuned", "rnn" or None
USE_SDE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_id = f"{ALGORITHM}_{ENVIRONMENT_NAME}"
model_id += "_sde" if USE_SDE else ""
model_id += f"_{REWARD_MODEL}" if REWARD_MODEL is not None else ""

script_path = Path(__file__).parent.resolve()
models_path = path.join(script_path, "models_final")
tensorboard_path = path.join(script_path, "..", "..", "rl_logs")
reward_model_path = path.join(
    script_path, "..", "reward_model", "models_final", f"{model_id}.ckpt"
)


# pylint: disable=too-few-public-methods
class CustomReward(RewardFn):
    """Custom reward based on fine-tuned reward model."""

    def __init__(self):
        """Initialize custom reward."""
        super().__init__()

        module = None

        if REWARD_MODEL == "mlp_single":
            module = LightningNetwork
        elif REWARD_MODEL == "rnn":
            module = LightningRNNNetwork
        else:
            module = LightningTrajectoryNetwork

        self.reward_model = module.load_from_checkpoint(
            reward_model_path, input_dim=17, hidden_dim=256, layer_num=12, output_dim=1
        )

    def __call__(
        self,
        state: np.ndarray,
        _action: np.ndarray,
        _next_state: np.ndarray,
        _done: np.ndarray,
    ) -> list:
        """Return reward given current state."""
        rewards = self.reward_model(torch.Tensor(state).to(DEVICE).unsqueeze(0))
        return [reward.detach().item() for reward in rewards]


def main():
    """Run RL agent training."""
    cpu_count = os.cpu_count()
    cpu_count = cpu_count if cpu_count is not None else 8

    # For PPO, the more environments there are, the more `num_timesteps`
    # shifts from `total_timesteps`
    env = make_vec_env(ENVIRONMENT_NAME, n_envs=cpu_count if ALGORITHM != "ppo" else 1)

    if REWARD_MODEL is not None:
        env = RewardVecEnvWrapper(env, reward_fn=CustomReward())

    if ALGORITHM == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_path,
            use_sde=USE_SDE,
        )
    elif ALGORITHM == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_path,
            use_sde=USE_SDE,
            device="cpu",
        )
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    iterations = 20
    steps_per_iteration = 25000
    timesteps = 0

    for i in range(iterations):
        trained_model = model.learn(
            total_timesteps=steps_per_iteration * (i + 1) - timesteps,
            reset_num_timesteps=False,
            tb_log_name=model_id,
        )

        timesteps = trained_model.num_timesteps

        model.save(path.join(models_path, f"{model_id}_{timesteps}"))


if __name__ == "__main__":
    main()
