"""Common types for all scripts in the project."""

from typing import Generic, Literal, TypedDict, TypeVar, Union, Tuple, List

import numpy
from numpy.typing import NDArray

# Different Feedback Types
FeedbackType = Union[
    Literal["evaluative"],
    Literal["comparative"],
    Literal["corrective"],
    Literal["demonstrative"],
    Literal["descriptive"],
    Literal["descriptive_preference"]
]

SegmentT = List[Tuple[NDArray, NDArray, bool, float]]

# Feedback Dataset
class FeedbackDataset(TypedDict):
    segments: List[SegmentT]
    ratings: List[int]
    preferences: List[Tuple[int, int, int]]
    demos: List[SegmentT]
    corrections: List[Tuple[SegmentT, SegmentT]]
    description: List[Tuple[NDArray, float]]
    description_preference: List[Tuple[int, int, int]]