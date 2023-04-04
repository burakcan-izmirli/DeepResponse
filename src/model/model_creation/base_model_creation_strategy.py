""" Base model creation strategy """
from abc import ABC, abstractmethod
from tensorflow import keras

from src.model.model_creation.mpnn import MessagePassing, TransformerEncoderReadout

from helper.enum.model.convolutional_model import ConvolutionalModel
from helper.enum.model.mlp_model import MLPModelDense, MLPModelDropout


class BaseModelCreationStrategy(ABC):
    """ Base model creation strategy """

    @abstractmethod
    def create_model(self, atom_dims, bond_dims, cell_line_dims, batch_size, message_units, message_steps,
                     num_attention_heads, dense_units):
        """ Create model """
        pass

    def create_mpnn_model(self, atom_dims, bond_dims, message_units, message_steps, num_attention_heads, dense_units,
                          batch_size):
        """
        Creating MPNN model
        """
        atom_features = keras.layers.Input((atom_dims), dtype="float32", name="atom_features")
        bond_features = keras.layers.Input((bond_dims), dtype="float32", name="bond_features")
        pair_indices = keras.layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = keras.layers.Input((), dtype="int32", name="molecule_indicator")

        x = MessagePassing(message_units, message_steps)([atom_features, bond_features, pair_indices])

        x = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)(
            [x, molecule_indicator])

        return atom_features, bond_features, pair_indices, molecule_indicator, x

    def create_conv_model(self, cell_line_dims):
        """
        Create convolutional neural network
        """
        input_layer = keras.layers.Input(shape=(cell_line_dims[1], cell_line_dims[2], 1))
        x = keras.layers.Conv2D(
            ConvolutionalModel.conv_1.filters, ConvolutionalModel.conv_1.kernel_size)(input_layer)
        x = keras.layers.Conv2D(
            ConvolutionalModel.conv_2.filters, ConvolutionalModel.conv_2.kernel_size)(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.GlobalAveragePooling2D()(x)

        return input_layer, x

    def create_mlp_model(self, dense_units, concat):
        """
        Create MLP model
        """
        x = keras.layers.Dense(dense_units, activation=MLPModelDense.dense_1.activation)(concat)
        x = keras.layers.Dropout(MLPModelDropout.dropout_1.rate)(x)
        x = keras.layers.Dense(MLPModelDense.dense_2.units, activation=MLPModelDense.dense_2.activation)(x)
        x = keras.layers.Dense(MLPModelDense.dense_3.units, activation=MLPModelDense.dense_3.activation)(x)

        return x
