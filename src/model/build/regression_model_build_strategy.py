""" Regression model strategy """
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import logging

from src.model.build.base_model_build_strategy import BaseModelCreationStrategy
from src.model.build.architecture.mlp_architecture import create_enhanced_mlp_model

class RegressionModelCreationStrategy(BaseModelCreationStrategy):
    """
    Regression model strategy for drug response prediction.
    """

    def create_model(self,
                     drug_input_shape,
                     cell_input_shape,
                     final_mlp_dense_units=1024,
                     selformer_trainable_layers=-1):
        """Create regression model for drug response prediction"""
        return super().create_model(
            drug_input_shape=drug_input_shape,
            cell_input_shape=cell_input_shape,
            final_mlp_dense_units=final_mlp_dense_units,
            selformer_trainable_layers=selformer_trainable_layers
        )

    def create_mlp_model(self, dense_units, input_tensor, prefix="regression_mlp"):
        """
        Create the MLP head for regression prediction.
        """
        return create_enhanced_mlp_model(dense_units, input_tensor, prefix)