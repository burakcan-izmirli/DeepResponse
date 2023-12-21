from enum import Enum


class MLPModelDense(Enum):
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    dense_1 = None, 'LeakyReLU'
    dense_2 = 128, 'LeakyReLU'
    dense_3 = 1, 'linear'


class MLPModelDropout(Enum):
    def __init__(self, rate):
        self.rate = rate

    dropout_1 = 0.2

