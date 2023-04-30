""" Merged model strategy """
from tensorflow import keras

from src.model.build.base_model_build_strategy import BaseModelCreationStrategy


class MergedModelStrategy(BaseModelCreationStrategy):
    """ Merged model strategy """

    def create_model(self, atom_dims, bond_dims, cell_line_dims, batch_size=32, message_units=64,
                     message_steps=4,
                     num_attention_heads=8,
                     dense_units=512):
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
