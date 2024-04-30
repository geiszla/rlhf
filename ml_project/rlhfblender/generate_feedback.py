"""Module for generating feedback data for a given RL agent using another, pretrained agent."""

import os
import pickle
import re
from copy import deepcopy
from os import path
from typing import Type, Union

import gym
import numpy
import torch
from captum.attr import IntegratedGradients
from numpy.typing import NDArray
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC
from torch import Tensor

from ..types import ActionNumpyT, Feedback, ObservationT, ObservationType
from .common import (
    ALGORITHM,
    ENVIRONMENT_NAME,
    MODEL_ID,
    STEPS_PER_CHECKPOINT,
    checkpoints_path,
    script_path,
)

TARGET_NEURON = 0

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


def get_attributions(
    observations: ObservationType,
    model: PPO,
    explainer: IntegratedGradients,
) -> NDArray[numpy.float64]:
    """
    Compute attributions for a given set of observations using the provided model and explainer.

    Args:
        observations (ObservationType): The input observations for which to compute attributions.
        model (PPO): The model used for prediction.
        explainer (IntegratedGradients): The explainer used to compute attributions.

    Returns:
        NDArray[numpy.float32]: The computed attributions.

    Raises:
        ValueError: If the observations tensor or observations baseline tensor is not
        a torch.Tensor.
    """
    observations = numpy.array(observations)
    observations_input = model.policy.obs_to_tensor(observations)[0]

    if not isinstance(observations_input, Tensor):
        raise ValueError("Observations tensor is not a torch.Tensor")

    observations_baseline = model.policy.obs_to_tensor(
        numpy.repeat(numpy.mean(observations, axis=0), observations.shape[0], axis=0)
    )[0]

    if not isinstance(observations_baseline, Tensor):
        raise ValueError("Observations baseline tensor is not a torch.Tensor")

    attribution = [
        (
            explainer.attribute(
                observations_input,
                # TODO: target set of neurons
                target=TARGET_NEURON,
                baselines=observations_baseline,
                internal_batch_size=64,
            )
            .cpu()
            .detach()
            .numpy()
        )
        for _ in range(0, observations.shape[0], 64)
    ]

    attribution = numpy.concatenate(attribution, axis=0)

    return attribution


def generate_feedback(
    model_class: Union[Type[PPO], Type[SAC]],
    environment: gym.Env[ObservationT, NDArray[ActionNumpyT]],
):
    """Generate agent's observations and feedback on them in the training environment."""
    feedback: list[Feedback[ObservationT, ActionNumpyT]] = []
    model_count = 0

    for file in os.listdir(checkpoints_path):
        if not re.search(f"{MODEL_ID}_[0-9]", file):
            continue

        # Load current agent from checkpoint file
        model = model_class.load(
            path.join(checkpoints_path, file[:-4]),
            custom_objects={"lr_schedule": lambda _: 0.0},
        )

        explainer = IntegratedGradients(
            # pylint: disable=cell-var-from-loop
            # TODO: test if this is correct
            lambda observations: expert_model.policy.value_net(
                expert_model.policy.mlp_extractor(
                    expert_model.policy.extract_features(observations)
                )[0]
            )
        )

        observations, _ = environment.reset()

        for _ in range(STEPS_PER_CHECKPOINT):
            # Get predicted actions and observations/rewards after taking the action
            action, _states = model.predict(observations, deterministic=True)

            # Note: his is very slow, might worth targetting mujoco and
            # using env.set_state instead
            # See https://github.com/openai/mujoco-py/blob/master/examples/setting_state.py
            environment_copy = deepcopy(environment)

            # Get value from value function of the expert
            observation_array = expert_model.policy.obs_to_tensor(
                numpy.array(observations)
            )[0]

            with torch.no_grad():
                expert_value = expert_model.policy.predict_values(observation_array)[0]
                expert_action, _states = expert_model.predict(
                    observations, deterministic=True
                )

            # Take the expert's action
            expert_observation, reward, terminated, _truncated, _info = (
                environment.step(expert_action)
            )

            # Restore environment state, then take the agent's action
            environment = environment_copy
            observations, reward, terminated, _truncated, _info = environment.step(
                action
            )

            # Add feedback to the list
            feedback.append(
                {
                    "action": action,
                    "observations": observations,
                    "reward": reward,
                    "expert_value": expert_value.item(),
                    "expert_action": expert_action,
                    "expert_observations": expert_observation,
                    "expert_value_attributions": get_attributions(
                        expert_observation, expert_model, explainer
                    ),
                }
            )

            if terminated:
                observations, _ = environment.reset()

        model_count += 1
        print(f"Model #{model_count}")

        # if model_count == 3:
        #     break

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
