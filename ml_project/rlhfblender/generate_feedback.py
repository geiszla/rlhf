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
    Compute attributions for a given observation and actions using the provided explainer.

    Args:
        observation (Tensor): The input observation for which to compute attributions.
        actions (NDArray[ActionType]): The action taken by the agent.
        explainer (IntegratedGradients): The explainer used to compute attributions.

    Returns:
        NDArray[numpy.float32]: The computed attributions.
    """
    observation_baselines = (
        torch.mean(observation).repeat(observation.shape[-1]).unsqueeze(0)
    )

    actions_tensor = torch.from_numpy(actions).unsqueeze(0).to(DEVICE)
    # TODO: change baseline to be 0
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

    print("Model ID:", MODEL_ID)

    for file in os.listdir(checkpoints_path):
        if not re.search(f"{MODEL_ID}_[0-9]", file):
            continue

        # Load current agent from checkpoint file
        model = model_class.load(
            path.join(checkpoints_path, file[:-4]),
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
        )

        # PPO
        # Note: This only works if `share_features_extractor` is True
        # explainer = IntegratedGradients(
        #     lambda observation, _actions: expert_model.policy.value_net(
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

        previous_expert_value = 0

        observation, _ = environment.reset()

        for _ in range(STEPS_PER_CHECKPOINT):
            # Save the current state of the environment before taking the expert's action
            # Note: only works with MuJoCo environments
            # (needs `sim.get_state()` and `sim.set_state()`)
            state_copy = environment.sim.get_state()  # type: ignore

            # Predict and take the expert's action
            expert_actions, _state = expert_model.predict(
                observation, deterministic=True
            )

            expert_observation, reward, terminated, _truncated, _info = (
                environment.step(expert_actions)
            )

            # Restore environment state, then predict and take the agent's action
            environment.sim.set_state(state_copy)  # type: ignore

            actions, _state = model.predict(observation, deterministic=True)
            next_observation, reward, terminated, _truncated, _info = environment.step(
                actions
            )

            # Predict expert value and observation/action attributions
            observation_tensor = expert_model.policy.obs_to_tensor(
                numpy.array(observation)
            )[0]

            assert isinstance(observation_tensor, Tensor)

            # PPO
            # next_observation_tensor = expert_model.policy.obs_to_tensor(
            #     numpy.array(next_observation)
            # )[0]

            # assert isinstance(next_observation_tensor, Tensor)

            # with torch.no_grad():
            #     expert_value = expert_model.policy.predict_values(next_observation_tensor)[0]

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

            # Add feedback to the list
            feedback.append(
                {
                    "actions": actions,
                    "observation": observation,
                    "reward": reward,
                    "expert_value": expert_value.item(),
                    "expert_value_difference": expert_value.item()
                    - previous_expert_value,
                    "expert_actions": expert_actions,
                    "expert_observation": expert_observation,
                    "expert_value_attributions": get_attributions(
                        observation_tensor, actions, explainer
                    ),
                }
            )

            previous_expert_value = expert_value.item()

            if terminated:
                observation, _ = environment.reset()
            else:
                observation = next_observation

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
