from enum import Enum


class DefaultArguments(Enum):
    random_state: int = 12
    batch_size: int = 64
    epoch: int = 50
    learning_rate: float = 0.01
    data_type: str = 'pathway'
    split_type: str = 'random'
    comet: bool = False
