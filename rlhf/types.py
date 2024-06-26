"""Common types for all scripts in the project."""

from typing import Generic, Literal, TypedDict, TypeVar, Union

import numpy
from numpy.typing import NDArray

FeedbackType = Union[
    Literal["evaluative"],
    Literal["comparative"],
    Literal["corrective"],
    Literal["demonstrative"],
    Literal["descriptive"],
]

ObservationType = Union[numpy.ndarray, dict[str, numpy.ndarray]]
ObservationT = TypeVar("ObservationT", bound=ObservationType)
ActionNumpyT = TypeVar("ActionNumpyT", bound=numpy.generic)


class Feedback(TypedDict, Generic[ObservationT, ActionNumpyT]):
    """Type for the generated feedback."""

    actions: NDArray[ActionNumpyT]
    observation: ObservationT
    next_observation: ObservationT
    reward: float

    expert_value: float
    expert_value_difference: float
    expert_observation: ObservationT
    expert_actions: NDArray[ActionNumpyT]
    # next_expert_observation: ObservationT
    expert_value_attributions: NDArray[numpy.float64]
    expert_own_value: float
