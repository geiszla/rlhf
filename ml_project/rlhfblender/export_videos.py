"""Module for saving videos and data of an RL agent's trajectories."""

from os import path
from pathlib import Path
from typing import Type, Union

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from .common import (
    ALGORITHM,
    ENVIRONMENT_NAME,
    MODEL_ID,
    checkpoints_path,
    get_reward_model_name,
)

RECORD_INTERVAL = 500
RECORD_LENGTH = 100
VIDEOS_PER_CHECKPOINT = 2

script_path = Path(__file__).parent.resolve()
reward_model_path = path.join(checkpoints_path, f"{get_reward_model_name(3653)}_500000")


def record_videos(
    model_class: Union[Type[PPO], Type[SAC]],
    environment: RecordVideo,
):
    """Record videos of the training environment."""
    model = model_class.load(reward_model_path)

    observations, _ = environment.reset()

    for _ in range(0, VIDEOS_PER_CHECKPOINT * RECORD_INTERVAL):
        action, _states = model.predict(observations, deterministic=True) # type: ignore
        observations, _reward, terminated, _truncated, _info = environment.step(action)

        environment.render()

        if terminated:
            observations = environment.reset()


def main():
    """Run video generation."""
    environment = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array")

    environment = RecordVideo(
        environment,
        video_folder=path.join(script_path, "..", "static", "videos"),
        step_trigger=lambda n: n % RECORD_INTERVAL == 0,
        video_length=RECORD_LENGTH,
        name_prefix=f"{MODEL_ID}",
    )

    if ALGORITHM == "sac":
        model_class = SAC
    elif ALGORITHM == "ppo":
        model_class = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    record_videos(model_class, environment)


if __name__ == "__main__":
    main()
