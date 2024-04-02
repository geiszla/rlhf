"""Module for generating feedback data for a given RL agent using another, pretrained agent."""

import os
import pickle
import re
from os import path
from typing import Type, Union

import gym
import numpy
import torch
from numpy.typing import NDArray
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC

from ..types import ActionNumpyT, Feedback, ObservationT
from .common import (
    ALGORITHM,
    ENVIRONMENT_NAME,
    MODEL_ID,
    STEPS_PER_CHECKPOINT,
    checkpoints_path,
    script_path,
)

# Load the pretrained "expert" model
expert_model = PPO.load(
    path.join(
        script_path, "..", "..", "logs", "ppo", "HalfCheetah-v3_1", "HalfCheetah-v3.zip"
    ),
    custom_objects={
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    },
)


def generate_feedback(
    algorithm: Union[Type[PPO], Type[SAC]],
    environment: gym.Env[ObservationT, NDArray[ActionNumpyT]],
):
    """Generate agent's observations and feedback on them in the training environment."""
    feedback: list[Feedback[ObservationT, ActionNumpyT]] = []
    model_count = 0

    for file in os.listdir(checkpoints_path):
        if not re.search(f"{MODEL_ID}_[0-9]", file):
            continue

        # Load current agent from checkpoint file
        model = algorithm.load(
            path.join(checkpoints_path, file[:-4]),
            custom_objects={"lr_schedule": lambda _: 0.0},
        )

        observations, _ = environment.reset()

        for _ in range(STEPS_PER_CHECKPOINT):
            # Get predicted actions and observations/rewards after taking the action
            action, _states = model.predict(observations, deterministic=True)
            observations, reward, terminated, _truncated, _info = environment.step(
                action
            )

            # Get value from value function of the expert
            observation_array = expert_model.policy.obs_to_tensor(
                numpy.array(observations)
            )[0]

            with torch.no_grad():
                expert_value = expert_model.policy.predict_values(observation_array)[0]
                expert_action, _states = expert_model.predict(
                    observations, deterministic=True
                )

            # Add feedback to the list
            feedback.append(
                {
                    "action": action,
                    "observations": observations,
                    "reward": reward,
                    "expert_value": expert_value.item(),
                    "expert_action": expert_action,
                }
            )

            if terminated:
                observations, _ = environment.reset()

        model_count += 1
        print(f"Model #{model_count}")

        if model_count == 3:
            break

    return feedback


def main():
    """Run data generation."""
    env = gym.make(ENVIRONMENT_NAME)

    # Select agent algorithm
    if ALGORITHM == "sac":
        model_class = SAC
    elif ALGORITHM == "ppo":
        model_class = PPO
    else:
        raise NotImplementedError(f"{ALGORITHM} not implemented")

    # Generate feedback
    feedback = generate_feedback(model_class, env)

    # Save feedback
    with open(
        path.join(script_path, "feedback", f"{MODEL_ID}.pkl"),
        "wb",
    ) as feedback_file:
        pickle.dump(feedback, feedback_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
