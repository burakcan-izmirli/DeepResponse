from enum import Enum


class MLPModelDense(Enum):
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    dense_1 = None, 'relu'
    dense_2 = 128, 'relu'
    dense_3 = 1, 'linear'


class MLPModelDropout(Enum):
    def __init__(self, rate):
        self.rate = rate

    dropout_1 = 0.2

