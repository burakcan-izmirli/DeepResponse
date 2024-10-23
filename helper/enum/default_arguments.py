from enum import Enum


class DefaultArguments(Enum):
    random_state: int = 12
    batch_size: int = 64
    epoch: int = 50
    learning_rate: float = 0.001
    data_source: str 'depmap'
    data_type: str = 'normal'
    split_type: str = 'random'
    learning_task: str = 'classification'
    comet: bool = False
