"""Script to plot the reward model output against the ground truth values."""

import pickle
from os import path
from pathlib import Path

import matplotlib
import numpy
import torch
from matplotlib import pyplot
from torch import Tensor

from ..reward_model.networks_old import LightningNetwork
from ..types import Feedback
from .common import DEVICE, FEEDBACK_ID, STEPS_PER_CHECKPOINT, get_reward_model_name

# Uncomment line below to use PyPlot with VSCode Tunnels
matplotlib.use("agg")

CHECKPOINT_NUMBER = 5
STEP_COUNT = 1000

REWARD_MODEL_ID = get_reward_model_name("8393_filtered")

script_path = Path(__file__).parent.resolve()

feedback_path = path.join(script_path, "feedback", f"{FEEDBACK_ID}.pkl")
reward_model_path = path.join(
    script_path, "reward_model_checkpoints", f"{REWARD_MODEL_ID}.ckpt"
)

output_path = path.join(script_path, "results", "reward_model_output.png")


def main():
    """Plot reward model output."""

    print("Feedback ID:", FEEDBACK_ID)
    print("Model ID:", REWARD_MODEL_ID)
    print()

    with open(feedback_path, "rb") as feedback_file:
        feedback_list: list[Feedback] = pickle.load(feedback_file)

    # pylint: disable=no-value-for-parameter
    reward_model = LightningNetwork.load_from_checkpoint(reward_model_path)

    feedback_start = STEPS_PER_CHECKPOINT * CHECKPOINT_NUMBER
    feedback_end = feedback_start + STEP_COUNT

    observations = list(map(lambda feedback: feedback["observation"], feedback_list))[
        feedback_start:feedback_end
    ]

    actions = list(map(lambda feedback: feedback["actions"], feedback_list))[
        feedback_start:feedback_end
    ]

    rewards = list(map(lambda feedback: feedback["reward"], feedback_list))[
        feedback_start:feedback_end
    ]

    expert_value_predictions = list(
        map(lambda feedback: feedback["expert_value"], feedback_list)
    )[feedback_start:feedback_end]

    predicted_rewards = []

    steps = range(STEP_COUNT)

    observation_tensor = Tensor(numpy.array(observations)).to(DEVICE)
    actions_tensor = Tensor(numpy.array(actions)).to(DEVICE)

    print("Predicting rewards...")

    for i in steps:
        predicted_rewards.append(
            reward_model(torch.cat([observation_tensor[i], actions_tensor[i]]))
            .detach()
            .cpu()
        )

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{STEP_COUNT} done")
            print(
                f"difference: {predicted_rewards[-1] - expert_value_predictions[i]}\n"
            )

    print()

    pyplot.plot(steps, predicted_rewards, label="Reward model")
    # pyplot.plot(steps, expert_value_predictions, label="Expert value")
    pyplot.plot(steps, rewards, label="Ground truth rewards")

    pyplot.xlabel("Steps")
    pyplot.ylabel("Rewards")
    pyplot.legend()

    pyplot.savefig(output_path)


if __name__ == "__main__":
    main()
