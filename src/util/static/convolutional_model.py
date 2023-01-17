from enum import Enum


class ConvolutionalModel(Enum):
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size

    conv_1 = 1, (4, 1)
    conv_2 = 1, (4, 1)

