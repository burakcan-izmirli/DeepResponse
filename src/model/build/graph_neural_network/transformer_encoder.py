"""
Transformer encoder
This code was derived from Keras' Message-passing neural network (MPNN) for the molecular property prediction tutorial
and extended based on the needs.
"""
import tensorflow as tf

from src.model.build.graph_neural_network.partition_padding import PartitionPadding


class TransformerEncoder(tf.keras.layers.Layer):
    """ Transformer encoder """

    def __init__(self, num_heads, embed_dim, dense_dim, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [tf.keras.layers.Dense(dense_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ]
        )
        self.normalization_layer_1 = tf.keras.layers.LayerNormalization()
        self.normalization_layer_2 = tf.keras.layers.LayerNormalization()
        self.average_pooling_layer = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs):
        """ Call """
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.normalization_layer_1(x + attention_output)
        proj_output = self.normalization_layer_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling_layer(proj_output)
