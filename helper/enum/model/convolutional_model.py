from enum import Enum


class ConvolutionalModel(Enum):
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size

    conv_1 = 32, (8, 1)  # Increase the number of filters from 1 to 32 and kernel size to (8, 1)
    conv_2 = 64, (4, 1)  # Increase the number of filters from 1 to 64
    conv_3 = 128, (4, 1)  # New convolutional layer
