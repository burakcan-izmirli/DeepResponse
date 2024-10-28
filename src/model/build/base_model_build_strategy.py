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
        Create graph neural network
        """
        atom_features = keras.layers.Input((atom_dims), dtype="float32", name="atom_features")
        bond_features = keras.layers.Input((bond_dims), dtype="float32", name="bond_features")
        pair_indices = keras.layers.Input((2), dtype="int32", name="pair_indices")
        molecule_indicator = keras.layers.Input((), dtype="int32", name="molecule_indicator")

        # Message passing with increased units and steps
        x = MessagePassing(message_units, message_steps)([atom_features, bond_features, pair_indices])

        # Apply Transformer encoder with batch normalization
        x = TransformerEncoder(num_attention_heads, message_units, dense_units, batch_size)([x, molecule_indicator])

        return atom_features, bond_features, pair_indices, molecule_indicator, x

    def create_conv_model(self, cell_line_dims):
        """
        Create convolutional neural network with batch normalization after each layer,
        using ConvolutionalModel enum for filters and kernel sizes.
        """
        input_layer = keras.layers.Input(shape=(cell_line_dims[1], cell_line_dims[2], 1))

        # First convolutional layer with batch normalization
        x = keras.layers.Conv2D(ConvolutionalModel.conv_1.filters, ConvolutionalModel.conv_1.kernel_size)(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(ConvolutionalModel.conv_1.activation)(x)

        # Second convolutional layer with batch normalization
        x = keras.layers.Conv2D(ConvolutionalModel.conv_2.filters, ConvolutionalModel.conv_2.kernel_size)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(ConvolutionalModel.conv_2.activation)(x)

        # Global average pooling
        x = keras.layers.GlobalAveragePooling2D()(x)

        return input_layer, x

    def create_model(self, atom_dims, bond_dims, cell_line_dims, batch_size=32, message_units=128,
                        message_steps=5,
                        num_attention_heads=8,
                        dense_units=1024):
            """
            Create merged model
            :param atom_dims: Atom dimensions
            :param bond_dims: Bond dimensions
            :param cell_line_dims: Cell line dimensions
            :param batch_size: Batch size
            :param message_units: Message units
            :param message_steps: Message steps
            :param num_attention_heads: Number of attention heads
            :param dense_units: Dense units
            :return: Merged model
            """
            atom_features, bond_features, pair_indices, molecule_indicator, mpnn = \
                self.create_graph_neural_network(atom_dims,
                                                bond_dims,
                                                message_units,
                                                message_steps,
                                                num_attention_heads,
                                                dense_units,
                                                batch_size)
            cell_line_features, conv = self.create_conv_model(cell_line_dims)

            concat = keras.layers.concatenate([conv, mpnn], name='concat')

            mlp = self.create_mlp_model(dense_units, concat)

            model_func = keras.Model(inputs=[cell_line_features, atom_features, bond_features, pair_indices,
                                            molecule_indicator],
                                    outputs=[mlp],
                                    name='final_output')

            return model_func

