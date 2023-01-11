from enum import Enum


class DefaultArguments(Enum):
    seed: int = 12
    batch_size: int = 64
    epoch: int = 50
    learning_rate: float = 0.01
    data_type: str = 'pathway'
