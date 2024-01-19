"""
Message passing
This code was derived from Keras' Message-passing neural network (MPNN) for the molecular property prediction tutorial
and extended based on the needs.
"""
import tensorflow as tf
from src.model.build.graph_neural_network.edge_network import EdgeNetwork


class MessagePassing(tf.keras.layers.Layer):
    """ Message passing """

    def __init__(self, units, steps, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.atom_dim = None
        self.message_step = None
        self.pad_length = None
        self.update_step = None
        self.built = None

    def build(self, input_shape):
        """ Build message passing """
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = tf.keras.layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        """ Call """
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated
