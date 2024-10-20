from enum import Enum


class ConvolutionalModel(Enum):
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size

    conv_1 = 64, (5, 1)
    conv_2 = 128, (5, 1)

