"""
MLP architectures for drug-cell line feature fusion and prediction.
"""
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow as tf
import logging

def create_enhanced_mlp_model(dense_units, input_tensor, prefix="enhanced_mlp"):
    """
    Enhanced MLP head with:
    - Multi-layer perceptron with residual connections
    - Self-attention for feature refinement
    - Highway networks for better gradient flow
    - Advanced regularization techniques
    """
    logging.info(f"Creating Enhanced MLP head ({prefix}) with {dense_units} units.")
    
    # Initial feature transformation
    x = layers.Dense(dense_units, activation='relu', 
                     kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_dense1')(input_tensor)
    x = layers.BatchNormalization(name=f'{prefix}_bn1')(x)
    x = layers.Dropout(0.3, name=f'{prefix}_dropout1')(x)
    
    # First residual block
    residual1 = layers.Dense(dense_units, activation='relu',
                            kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_residual1_dense1')(x)
    residual1 = layers.BatchNormalization(name=f'{prefix}_residual1_bn1')(residual1)
    residual1 = layers.Dropout(0.2, name=f'{prefix}_residual1_dropout')(residual1)
    residual1 = layers.Dense(dense_units, activation='linear',
                            kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_residual1_dense2')(residual1)
    
    # Add residual connection
    x = layers.Add(name=f'{prefix}_residual1_add')([x, residual1])
    x = layers.Activation('relu', name=f'{prefix}_residual1_activation')(x)
    x = layers.BatchNormalization(name=f'{prefix}_residual1_bn2')(x)
    
    # Highway network layer
    highway_units = dense_units // 2
    highway_transform = layers.Dense(highway_units, activation='sigmoid',
                                   kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_highway_transform')(x)
    highway_carry = layers.Lambda(lambda x: 1.0 - x, name=f'{prefix}_highway_carry')(highway_transform)
    
    highway_features = layers.Dense(highway_units, activation='relu',
                                  kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_highway_features')(x)
    highway_projected = layers.Dense(highway_units, activation='linear', name=f'{prefix}_highway_projection')(x)
    
    # Highway connection
    x = layers.Add(name=f'{prefix}_highway_add')([
        layers.Multiply(name=f'{prefix}_highway_transform_mult')([highway_features, highway_transform]),
        layers.Multiply(name=f'{prefix}_highway_carry_mult')([highway_projected, highway_carry])
    ])
    x = layers.BatchNormalization(name=f'{prefix}_highway_bn')(x)
    
    # Self-attention layer for feature refinement
    attention_dim = highway_units
    attention_query = layers.Dense(attention_dim, name=f'{prefix}_attention_query')(x)
    attention_key = layers.Dense(attention_dim, name=f'{prefix}_attention_key')(x)
    attention_value = layers.Dense(attention_dim, name=f'{prefix}_attention_value')(x)
    
    # Reshape for attention computation
    query = layers.Reshape((1, attention_dim), name=f'{prefix}_query_reshape')(attention_query)
    key = layers.Reshape((1, attention_dim), name=f'{prefix}_key_reshape')(attention_key)
    value = layers.Reshape((1, attention_dim), name=f'{prefix}_value_reshape')(attention_value)
    
    # Compute attention scores
    attention_scores = layers.Dot(axes=-1, name=f'{prefix}_attention_scores')([query, key])
    attention_scores = layers.Lambda(lambda x: x / tf.sqrt(float(attention_dim)), name=f'{prefix}_attention_scale')(attention_scores)
    attention_weights = layers.Softmax(name=f'{prefix}_attention_weights')(attention_scores)
    
    # Apply attention
    attended_features = layers.Dot(axes=1, name=f'{prefix}_attention_output')([attention_weights, value])
    attended_features = layers.Reshape((attention_dim,), name=f'{prefix}_attention_flatten')(attended_features)
    
    # Combine original and attended features
    x = layers.Add(name=f'{prefix}_attention_residual')([x, attended_features])
    x = layers.LayerNormalization(name=f'{prefix}_attention_norm')(x)
    
    # Second residual block
    residual_units = highway_units // 2
    residual2 = layers.Dense(residual_units, activation='relu',
                            kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_residual2_dense1')(x)
    residual2 = layers.BatchNormalization(name=f'{prefix}_residual2_bn1')(residual2)
    residual2 = layers.Dropout(0.2, name=f'{prefix}_residual2_dropout')(residual2)
    residual2 = layers.Dense(residual_units, activation='linear',
                            kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_residual2_dense2')(residual2)
    
    # Project x to same dimension for residual connection
    x_projected = layers.Dense(residual_units, activation='linear', name=f'{prefix}_residual2_projection')(x)
    x = layers.Add(name=f'{prefix}_residual2_add')([x_projected, residual2])
    x = layers.Activation('relu', name=f'{prefix}_residual2_activation')(x)
    x = layers.BatchNormalization(name=f'{prefix}_residual2_bn2')(x)
    
    # Final prediction layers
    final_units = max(64, residual_units // 2)
    x = layers.Dense(final_units, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name=f'{prefix}_final_dense1')(x)
    x = layers.BatchNormalization(name=f'{prefix}_final_bn')(x)
    x = layers.Dropout(0.2, name=f'{prefix}_final_dropout')(x)
    
    # Output layer
    output = layers.Dense(1, activation='linear', name=f'{prefix}_output_layer')(x)
    
    return output


def create_cross_attention_fusion(drug_features, cell_features, output_dim=512, prefix="cross_attention"):
    """
    Cross-attention fusion between drug and cell line features.
    """    
    # Get dimensions
    drug_dim = drug_features.shape[-1]
    cell_dim = cell_features.shape[-1]
    
    # Project to same dimension
    drug_proj = layers.Dense(output_dim, activation='relu', name=f'{prefix}_drug_projection')(drug_features)
    cell_proj = layers.Dense(output_dim, activation='relu', name=f'{prefix}_cell_projection')(cell_features)
    
    # Drug attending to cell features
    drug_query = layers.Dense(output_dim, name=f'{prefix}_drug_query')(drug_proj)
    cell_key = layers.Dense(output_dim, name=f'{prefix}_cell_key')(cell_proj)
    cell_value = layers.Dense(output_dim, name=f'{prefix}_cell_value')(cell_proj)
    
    # Reshape for attention
    drug_query = layers.Reshape((1, output_dim))(drug_query)
    cell_key = layers.Reshape((1, output_dim))(cell_key)
    cell_value = layers.Reshape((1, output_dim))(cell_value)
    
    # Compute attention scores
    drug_to_cell_scores = layers.Dot(axes=-1)([drug_query, cell_key])
    drug_to_cell_scores = layers.Lambda(lambda x: x / tf.sqrt(float(output_dim)))(drug_to_cell_scores)
    drug_to_cell_weights = layers.Softmax()(drug_to_cell_scores)
    
    # Apply attention
    drug_attended = layers.Dot(axes=1)([drug_to_cell_weights, cell_value])
    drug_attended = layers.Reshape((output_dim,))(drug_attended)
    
    # Cell attending to drug features
    cell_query = layers.Dense(output_dim, name=f'{prefix}_cell_query')(cell_proj)
    drug_key = layers.Dense(output_dim, name=f'{prefix}_drug_key')(drug_proj)
    drug_value = layers.Dense(output_dim, name=f'{prefix}_drug_value')(drug_proj)
    
    # Reshape for attention
    cell_query = layers.Reshape((1, output_dim))(cell_query)
    drug_key = layers.Reshape((1, output_dim))(drug_key)
    drug_value = layers.Reshape((1, output_dim))(drug_value)
    
    # Compute attention scores
    cell_to_drug_scores = layers.Dot(axes=-1)([cell_query, drug_key])
    cell_to_drug_scores = layers.Lambda(lambda x: x / tf.sqrt(float(output_dim)))(cell_to_drug_scores)
    cell_to_drug_weights = layers.Softmax()(cell_to_drug_scores)
    
    # Apply attention
    cell_attended = layers.Dot(axes=1)([cell_to_drug_weights, drug_value])
    cell_attended = layers.Reshape((output_dim,))(cell_attended)
    
    # Combine attended features
    fused_features = layers.Concatenate(name=f'{prefix}_concat')([
        drug_proj, cell_proj,           # Original features
        drug_attended, cell_attended    # Cross-attended features
    ])
    
    # Final fusion layer
    fused = layers.Dense(output_dim, activation='relu', name=f'{prefix}_fusion_dense')(fused_features)
    fused = layers.LayerNormalization(name=f'{prefix}_fusion_norm')(fused)
    
    return fused