""" Regression model strategy """
from tensorflow import keras

from src.model.build.base_model_build_strategy import BaseModelCreationStrategy


class RegressionModelCreationStrategy(BaseModelCreationStrategy):
    """ Regression model strategy """

    def create_mlp_model(self, dense_units, concat):
        """
        Create MLP model
        """
        x = keras.layers.Dense(dense_units, activation=MLPModelDense.dense_1.activation)(concat)
        x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Dropout(MLPModelDropout.dropout_1.rate)(x)

        x = keras.layers.Dense(MLPModelDense.dense_2.units, activation=MLPModelDense.dense_2.activation)(x)
        x = keras.layers.BatchNormalization()(x)
        # x = keras.layers.Dropout(MLPModelDropout.dropout_2.rate)(x)

        x = keras.layers.Dense(MLPModelDense.dense_3.units)(x)

        return x

    