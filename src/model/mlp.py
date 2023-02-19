"""
Multilayer perceptron model
"""
from tensorflow.keras import layers

from src.util.enum.mlp_model import MLPModelDense, MLPModelDropout


def create_mlp_model(dense_units, concat):
    """
    Create MLP model
    """
    x = layers.Dense(dense_units, activation=MLPModelDense.dense_1.activation)(concat)
    x = layers.Dropout(MLPModelDropout.dropout_1.rate)(x)
    x = layers.Dense(MLPModelDense.dense_2.units, activation=MLPModelDense.dense_2.activation)(x)
    x = layers.Dense(MLPModelDense.dense_3.units, activation=MLPModelDense.dense_3.activation)(x)

    return x
