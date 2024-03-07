"""Module for training a reward model from the generated feedback."""

import pickle
from os import path
from pathlib import Path

from ..types import Feedback
from .common import MODEL_ID

script_path = Path(__file__).parent.resolve()


def main():
    """Run reward model training using generated feedback."""

    with open(
        path.join(
            script_path,
            "feedback",
            f"{MODEL_ID}.pkl",
        ),
        "rb",
    ) as feedback_file:
        feedback: list[Feedback] = pickle.load(feedback_file)

    print(feedback[0])
    print(feedback[1])
    print(feedback[2])


if __name__ == "__main__":
    main()
