from enum import Enum


class MLPModelDense(Enum):
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    dense_1 = 1024, 'relu'
    dense_2 = 512, 'relu'
    dense_3 = 1, ''


class MLPModelDropout(Enum):
    def __init__(self, rate):
        self.rate = rate

    dropout_1 = 0.2
    dropout_2 = 0.1


