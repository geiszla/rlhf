"""Module for generating feedback data for a given RL agent using another, pretrained agent."""

import os
import pickle
import re
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

from ..types import ActionNumpyT, Feedback, ObservationT
from .common import (
    ALGORITHM,
    DEVICE,
    ENVIRONMENT_NAME,
    MODEL_ID,
    STEPS_PER_CHECKPOINT,
    checkpoints_path,
    script_path,
)

# Load the pretrained "expert" model
# PPO
# expert_model = PPO.load(
#     path.join(
#         script_path, "..", "..", "logs", "ppo", "HalfCheetah-v3_1", "HalfCheetah-v3.zip"
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
        script_path, "..", "..", "logs", "sac", "HalfCheetah-v3_1", "HalfCheetah-v3.zip"
    ),
    custom_objects={
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    },
)


def get_attributions(
    observation: Tensor,
    actions: NDArray[ActionNumpyT],
    explainer: IntegratedGradients,
) -> NDArray[numpy.float64]:
    """
    Compute attributions for a given observation using the provided model and explainer.

    Args:
        observation (ObservationType): The input observation for which to compute attributions.
        model (PPO): The model used for prediction.
        explainer (IntegratedGradients): The explainer used to compute attributions.

    Returns:
        NDArray[numpy.float32]: The computed attributions.

    Raises:
        ValueError: If the observation tensor or observation baseline tensor is not
        a torch.Tensor.
    """
    observation_baselines = (
        torch.mean(observation).repeat(observation.shape[-1]).unsqueeze(0)
    )

    actions_tensor = torch.from_numpy(actions).unsqueeze(0).to(DEVICE)
    actions_baselines = (
        torch.mean(actions_tensor).repeat(actions.shape[-1]).unsqueeze(0)
    )

    attribution = [
        torch.cat(
            explainer.attribute(
                (observation, actions_tensor),
                target=0,
                baselines=(observation_baselines, actions_baselines),
                internal_batch_size=64,
            ),
            dim=1,
        )
        for _ in range(0, observation.shape[0], 64)
    ]

    return torch.cat(attribution, dim=1).squeeze().cpu().numpy()


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

        # PPO
        # Note: This only works if `share_features_extractor` is True
        # explainer = IntegratedGradients(
        #     lambda observation: expert_model.policy.value_net(
        #         expert_model.policy.mlp_extractor(
        #             expert_model.policy.extract_features(observation)
        #         )[0]
        #     )
        # )

        # SAC
        explainer = IntegratedGradients(
            lambda observation, actions: torch.min(
                torch.cat(
                    expert_model.policy.critic_target(observation, actions),
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )[0]
        )

        observation, _ = environment.reset()

        for _ in range(STEPS_PER_CHECKPOINT):
            # Get predicted actions and observations/rewards after taking the action
            actions, _state = model.predict(observation, deterministic=True)

            # Note: only works with MuJoCo environments
            # (needs `sim.get_state()` and `sim.set_state()`)
            state_copy = environment.sim.get_state()  # type: ignore

            # Get value from value function of the expert
            observation_tensor = expert_model.policy.obs_to_tensor(
                numpy.array(observation)
            )[0]

            assert isinstance(observation_tensor, Tensor)

            # PPO
            # with torch.no_grad():
            #     expert_value = expert_model.policy.predict_values(observation_tensor)[0]

            # SAC
            expert_value = torch.min(
                torch.cat(
                    expert_model.policy.critic_target(
                        observation_tensor,
                        torch.from_numpy(actions).unsqueeze(0).to(DEVICE),
                    ),
                    dim=1,
                ),
                dim=1,
            )[0]

            expert_actions, _state = expert_model.predict(
                observation, deterministic=True
            )

            # Take the expert's action
            expert_observation, reward, terminated, _truncated, _info = (
                environment.step(expert_actions)
            )

            # Restore environment state, then take the agent's action
            environment.sim.set_state(state_copy)  # type: ignore
            observation, reward, terminated, _truncated, _info = environment.step(
                actions
            )

            # Add feedback to the list
            feedback.append(
                {
                    "actions": actions,
                    "observation": observation,
                    "reward": reward,
                    "expert_value": expert_value.item(),
                    "expert_actions": expert_actions,
                    "expert_observation": expert_observation,
                    "expert_value_attributions": get_attributions(
                        observation_tensor, expert_actions, explainer
                    ),
                }
            )

            if terminated:
                observation, _ = environment.reset()

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
