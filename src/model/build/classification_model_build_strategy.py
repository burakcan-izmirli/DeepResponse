""" Classification model strategy """
from tensorflow import keras
from src.model.build.base_model_build_strategy import BaseModelCreationStrategy

class ClassificationModelCreationStrategy(BaseModelCreationStrategy):
    """Classification model creation strategy."""

    def create_mlp_model(self, dense_units, concat, prefix="final_mlp"):
        """Creates the MLP head for classification."""
        x = keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(concat)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(dense_units // 2, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)

        output = keras.layers.Dense(1, activation='sigmoid', name=f"{prefix}_output")(x)
        return output