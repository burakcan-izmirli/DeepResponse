from enum import Enum


class DefaultArguments(Enum):
    random_state: int = 12
    batch_size: int = 32
    epoch: int = 100
    learning_rate: float = 0.0001
    data_source: str = 'depmap'
    data_type: str = 'normal'
    split_type: str = 'random'
    learning_task: str = 'regression'
    comet: bool = False
    selformer_trainable_layers: int = -1
