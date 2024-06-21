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

script_path = Path(__file__).parent.resolve()

reward_model_path = path.join(
    checkpoints_path, f"{get_reward_model_name('3161')}_3625008"
)

output_path = path.join(script_path, "..", "static", "videos")


def record_videos(
    model_class: Union[Type[PPO], Type[SAC]],
    environment: RecordVideo,
):
    """Record videos of the training environment."""
    model = model_class.load(reward_model_path)

    observation, _ = environment.reset()

    for _ in range(0, RECORD_INTERVAL * 2):
        actions, _states = model.predict(observation, deterministic=True)  # type: ignore
        observation, _reward, terminated, _truncated, _info = environment.step(actions)

        environment.render()

        if terminated:
            observation = environment.reset()


def main():
    """Run video generation."""
    print("Reward model path:", reward_model_path)
    print()

    environment = gym.make(ENVIRONMENT_NAME, render_mode="rgb_array")

    environment = RecordVideo(
        environment,
        video_folder=output_path,
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
