"""Common types for all scripts in the project."""

from typing import Generic, Literal, TypedDict, TypeVar, Union

import numpy
from numpy.typing import NDArray
from torch import Tensor

FloatNDArray = NDArray[numpy.float_]


class Step(TypedDict):
    """Trajectory dictionary type."""

    obs: FloatNDArray
    reward: float


Trajectory = list[Step]
Trajectories = dict[int, Trajectory]
Batch = list[tuple[list[FloatNDArray], list[FloatNDArray]]]
TensorBatch = list[tuple[Tensor, Tensor]]

Obs = FloatNDArray
RewardlessTrajectory = list[Obs]
RewardlessTrajectories = dict[int, RewardlessTrajectory]

# Feedback
ObservationType = Union[numpy.ndarray, dict[str, numpy.ndarray]]
ObservationT = TypeVar("ObservationT", bound=ObservationType)
ActionNumpyT = TypeVar("ActionNumpyT", bound=numpy.generic)


class Feedback(TypedDict, Generic[ObservationT, ActionNumpyT]):
    """Type for the generated feedback."""

    actions: NDArray[ActionNumpyT]
    observation: ObservationT
    reward: float

    expert_value: float
    expert_actions: NDArray[ActionNumpyT]
    expert_observation: ObservationT
    expert_value_attributions: NDArray[numpy.float64]


FeedbackType = Union[
    Literal["evaluative"],
    Literal["comparative"],
    Literal["corrective"],
    Literal["demonstrative"],
    Literal["descriptive"],
]
