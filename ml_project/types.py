"""Common types for all scripts in the project."""

from typing import Generic, TypedDict, TypeVar, Union

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
ObservationT = TypeVar(
    "ObservationT", bound=Union[numpy.ndarray, dict[str, numpy.ndarray]]
)
ActionNumpyT = TypeVar("ActionNumpyT", bound=numpy.generic)


class Feedback(TypedDict, Generic[ObservationT, ActionNumpyT]):
    """Type for the generated feedback."""

    action: NDArray[ActionNumpyT]
    observations: ObservationT
    reward: float

    expert_value: float
    expert_action: NDArray[ActionNumpyT]
