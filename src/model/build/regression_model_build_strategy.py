""" Regression model strategy """
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import logging

from src.model.build.base_model_build_strategy import BaseModelCreationStrategy

class RegressionModelCreationStrategy(BaseModelCreationStrategy):
    """
    Regression model strategy with:
    - Standard MLP head for regression tasks.
    """

    def create_mlp_model(self, dense_units, input_tensor, prefix="final_mlp"):
        """
        Create the MLP head for regression.
        """
        logging.info(f"Creating Regression MLP head ({prefix}) with {dense_units} units.")

        # First layer
        x = layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{prefix}_dense_1"
        )(input_tensor)
        x = layers.LayerNormalization(name=f"{prefix}_ln_1")(x)
        x = layers.Dropout(0.3, name=f"{prefix}_dropout_1")(x)

        # Second layer
        x = layers.Dense(
            dense_units // 2,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{prefix}_dense_2"
        )(x)
        x = layers.LayerNormalization(name=f"{prefix}_ln_2")(x)
        x = layers.Dropout(0.4, name=f"{prefix}_dropout_2")(x)

        # Output layer
        output = layers.Dense(
            1,
            activation='linear',
            name=f"{prefix}_output"
        )(x)

        return output