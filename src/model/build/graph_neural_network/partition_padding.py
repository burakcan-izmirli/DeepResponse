"""
Partition padding
This code was derived from Keras' Message-passing neural network (MPNN) for the molecular property prediction tutorial
and extended based on the needs.
"""
import tensorflow as tf


class PartitionPadding(tf.keras.layers.Layer):
    """ Partition padding """

    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        """ Call """
        atom_features, molecule_indicator = inputs

        # Obtain sub-graphs
        atom_features_partitioned = tf.dynamic_partition(atom_features, molecule_indicator, self.batch_size)

        # Pad and stack sub-graphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack([
            tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
            for f, n in zip(atom_features_partitioned, num_atoms)
        ])

        # Remove empty sub-graphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)
