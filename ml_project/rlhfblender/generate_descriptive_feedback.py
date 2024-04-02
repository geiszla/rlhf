"""Test script for generating descriptive feedback for a given RL agent."""

import time

import gym
import numpy
from captum.attr import IntegratedGradients
from stable_baselines3.ppo.ppo import PPO
from torch import Tensor

from .common import ENVIRONMENT_NAME, USE_SDE

# We need to target a single neuron, here neuron 0 which encodes the first action dimension
# If we have multi-dimensional actions, we may want to run the attribution computation for multiple
# logit neurons and average them

# Target the first neuron for now (later we can change this to a list of neurons)
TARGET_NEURON = 0


def main():
    """Run descriptive feedback generation."""

    environment = gym.make(ENVIRONMENT_NAME)

    model = PPO(
        "MlpPolicy",
        environment,  # type: ignore
        verbose=1,
        use_sde=USE_SDE,
    )

    explainer = IntegratedGradients(
        lambda observations: model.policy.action_net(
            model.policy.mlp_extractor(model.policy.extract_features(observations))[0]
        )
    )

    print("\nRunning with Explainer module...")

    start_time = time.time()

    observations, _ = environment.reset()
    action, _states = model.predict(observations, deterministic=True)
    observations, _reward, _terminated, _truncated, _info = environment.step(action)

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
    attribution[attribution < 0] = 0

    print(attribution)
    print(f"Finished generation of explanations in {time.time() - start_time:.2}s")


if __name__ == "__main__":
    main()
