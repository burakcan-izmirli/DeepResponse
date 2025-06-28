from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
import logging

from src.model.build.architecture.selformer_architecture import SELFormerLayer
from src.model.build.architecture.cnn_architecture import create_enhanced_conv_model
from src.model.build.architecture.mlp_architecture import create_enhanced_mlp_model

class BaseModelCreationStrategy(ABC):
    @abstractmethod
    def create_mlp_model(self, dense_units, input_tensor, prefix="final_mlp"):
        """ Abstract method for the final MLP head specific to the learning task (regression/classification)."""
        pass

    def create_selformer_network(self, drug_smiles_input_shape, num_trainable_encoder_layers=-1):
        drug_input = layers.Input(shape=drug_smiles_input_shape, dtype="string", name="drug_smiles_input")
        selformer_layer = SELFormerLayer(num_trainable_encoder_layers=num_trainable_encoder_layers)
        drug_embedding = selformer_layer(drug_input)
        return Model(inputs=drug_input, outputs=drug_embedding, name="selformer_network")

    def create_conv_model(self, cell_line_dims):
        """Create CNN model for cell line feature extraction with multi-scale convolutions and attention pooling"""
        return create_enhanced_conv_model(cell_line_dims)

    def create_model(self,
                     drug_input_shape,
                     cell_input_shape,
                     final_mlp_dense_units=1024,
                     selformer_trainable_layers=-1):

        if drug_input_shape is None or drug_input_shape != ():
             logging.warning(f"drug_input_shape was {drug_input_shape}, expected (). Overriding.")
             drug_input_shape = () # SELFormer expects scalar string input, shape ()
        if not cell_input_shape or len(cell_input_shape) != 2:
             raise ValueError(f"cell_input_shape for CNN part must be 2D, got {cell_input_shape}")

        logging.info(f"SELFormer Trainable Layers: {selformer_trainable_layers}")

        drug_network = self.create_selformer_network(drug_input_shape, selformer_trainable_layers)
        conv_network = self.create_conv_model(cell_input_shape)

        concatenated = layers.Concatenate(axis=1)([drug_network.output, conv_network.output])

        # Use MLP for final prediction
        final_output = create_enhanced_mlp_model(final_mlp_dense_units, concatenated)

        model = Model(inputs=[drug_network.input, conv_network.input], outputs=final_output)
        logging.info("Model created successfully.")
        return model