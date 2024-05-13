"""Script to plot the reward model output against the ground truth values."""

import pickle
from os import path
from pathlib import Path

import matplotlib
import numpy
from matplotlib import pyplot
from torch import Tensor

from ..reward_model.networks_old import LightningNetwork
from ..types import Feedback
from .common import DEVICE, MODEL_ID, STEPS_PER_CHECKPOINT, get_reward_model_name

# Uncomment line below to use PyPlot with VSCode Tunnels
matplotlib.use("agg")

script_path = Path(__file__).parent.resolve()

reward_model_path = path.join(
    script_path, "reward_model_checkpoints", f"{get_reward_model_name(3653)}.ckpt"
)

CHECKPOINT_NUMBER = 0


def main():
    """Plot reward model output."""

    feedback_path = path.join(script_path, "feedback", f"{MODEL_ID}.pkl")

    with open(feedback_path, "rb") as feedback_file:
        feedback_list: list[Feedback] = pickle.load(feedback_file)

    # pylint: disable=no-value-for-parameter
    reward_model = LightningNetwork.load_from_checkpoint(reward_model_path)

    feedback_start = STEPS_PER_CHECKPOINT * CHECKPOINT_NUMBER
    feedback_end = feedback_start + STEPS_PER_CHECKPOINT

    observations = list(map(lambda feedback: feedback["observations"], feedback_list))[
        feedback_start:feedback_end
    ]

    rewards = list(map(lambda feedback: feedback["reward"], feedback_list))[
        feedback_start:feedback_end
    ]

    expert_value_predictions = list(
        map(lambda feedback: feedback["expert_value"], feedback_list)
    )[feedback_start:feedback_end]

    predicted_rewards = []

    steps = range(STEPS_PER_CHECKPOINT)

    observations_tensor = Tensor(numpy.array(observations)).to(DEVICE)

    print("Predicting rewards...")

    for i in steps:
        predicted_rewards.append(reward_model(observations_tensor[i]).detach().cpu())

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{STEPS_PER_CHECKPOINT} done")
            print(
                f"difference: {predicted_rewards[-1] - expert_value_predictions[i]}\n"
            )

    print()

    pyplot.plot(steps, predicted_rewards, label="Predicted rewards")
    pyplot.plot(steps, expert_value_predictions, label="Predicted rewards (expert)")
    pyplot.plot(steps, rewards, label="Ground truth rewards")

    pyplot.xlabel("Steps")
    pyplot.ylabel("Rewards")
    pyplot.legend()

    pyplot.savefig(path.join(script_path, "results", "reward_model_output.png"))


if __name__ == "__main__":
    main()
