"""
Convolutional model
"""
from tensorflow.keras import layers

from src.util.static.convolutional_model import ConvolutionalModel


def create_conv_model(cell_line_dims):
    """
    Create convolutional neural network
    """
    input_layer = layers.Input(shape=(cell_line_dims[1], cell_line_dims[2], 1))
    x = layers.Conv2D(
        ConvolutionalModel.conv_1.filters, ConvolutionalModel.conv_1.kernel_size)(input_layer)
    x = layers.Conv2D(
        ConvolutionalModel.conv_2.filters, ConvolutionalModel.conv_2.kernel_size)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    return input_layer, x
