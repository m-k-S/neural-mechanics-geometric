import torch
from typing import Union
from dataclasses import dataclass

@dataclass
class InitializationMetric:
    layer_type: str
    layer_index: int
    depth: int
    normalization: str
    name: str
    value: float

    # Optional fields
    feature: Union[int, None] = None

@dataclass
class TrainingMetric:
    layer_type: Union[str, None] = None
    layer_index: Union[int, None] = None
    depth: int
    normalization: str
    name: str
    values: list

    # Training settings
    optimizer: str

    # Optional fields
    feature: Union[int, None] = None
