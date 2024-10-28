from enum import Enum


class ConvolutionalModel(Enum):
    def __init__(self, filters, kernel_size, activation):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

    conv_1 = 64, (5, 1), 'relu'
    conv_2 = 128, (5, 1), 'relu'

