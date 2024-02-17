""" Base model creation strategy """
from abc import ABC, abstractmethod
from tensorflow import keras

from src.model.build.graph_neural_network.message_passing import MessagePassing
from src.model.build.graph_neural_network.transformer_encoder import TransformerEncoder

from helper.enum.model.convolutional_model import ConvolutionalModel
from helper.enum.model.mlp_model import MLPModelDense, MLPModelDropout


class BaseModelCreationStrategy(ABC):
    """ Base model creation strategy """

    @abstractmethod
    def create_model(self, atom_dims, bond_dims, cell_line_dims, batch_size, message_units, message_steps,
                     num_attention_heads, dense_units):
        """ Create model """
        pass

    def create_graph_neural_network(self, atom_dims, bond_dims, message_units, message_steps, num_attention_heads,
                                    dense_units, batch_size):
        """
        Creating graph neural network
        """
        atom_features = keras.layers.Input((atom_dims), dtype="float32", name="atom_features")
        bond_features = keras.layers.Input((bond_dims), dtype="float32", name="bond_features")
        pair_indices = keras.layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = keras.layers.Input((), dtype="int32", name="molecule_indicator")

        x = MessagePassing(message_units, message_steps)([atom_features, bond_features, pair_indices])

        x = TransformerEncoder(num_attention_heads, message_units, dense_units, batch_size)([x, molecule_indicator])

        return atom_features, bond_features, pair_indices, molecule_indicator, x

    def create_conv_model(self, cell_line_dims):
        """
        Create convolutional neural network
        """
        input_layer = keras.layers.Input(shape=(cell_line_dims[1], cell_line_dims[2], 1))

        x = keras.layers.Conv2D(64, (1, 2), activation='relu', padding='same')(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 1))(x)

        # Convolutional Layer 2
        x = keras.layers.Conv2D(128, (1, 2), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 1))(x)

        # Convolutional Layer 3
        x = keras.layers.Conv2D(256, (1, 2), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 1))(x)

        # Convolutional Layer 4
        x = keras.layers.Conv2D(512, (1, 2), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 1))(x)

        # Flatten layer
        x = keras.layers.Flatten()(x)

        # Dense Layer
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)

        return input_layer, x

    def create_mlp_model(self, dense_units, concat):
        """
        Create MLP model
        """
        x = keras.layers.Dense(2048, activation='relu')(concat)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(1, activation='linear')(x)

        return x
