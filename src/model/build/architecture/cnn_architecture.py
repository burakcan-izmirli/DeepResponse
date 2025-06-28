"""
CNN architectures for cell line feature extraction.
"""
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import logging

def create_enhanced_conv_model(cell_line_dims):
    """
    Enhanced CNN architecture with:
    - Multiple Conv1D layers with different kernel sizes
    - Residual connections
    - Batch normalization and dropout
    - Multi-scale feature extraction
    - Attention pooling
    """
    input_shape_cnn = (cell_line_dims[0], cell_line_dims[1], 1)

    input_layer = layers.Input(shape=input_shape_cnn, name="cell_line_input")
    
    # Reshape for Conv1D processing
    x = layers.Reshape((cell_line_dims[0] * cell_line_dims[1], 1))(input_layer)
    
    # Multi-scale Conv1D feature extraction
    # Branch 1: Small receptive field (local patterns)
    branch1 = layers.Conv1D(64, 3, padding='same', activation='relu', 
                           kernel_regularizer=regularizers.l2(1e-5), name='conv1d_branch1_1')(x)
    branch1 = layers.BatchNormalization(name='bn_branch1_1')(branch1)
    branch1 = layers.Conv1D(64, 3, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(1e-5), name='conv1d_branch1_2')(branch1)
    branch1 = layers.BatchNormalization(name='bn_branch1_2')(branch1)
    
    # Branch 2: Medium receptive field (intermediate patterns)  
    branch2 = layers.Conv1D(64, 7, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(1e-5), name='conv1d_branch2_1')(x)
    branch2 = layers.BatchNormalization(name='bn_branch2_1')(branch2)
    branch2 = layers.Conv1D(64, 7, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(1e-5), name='conv1d_branch2_2')(branch2)
    branch2 = layers.BatchNormalization(name='bn_branch2_2')(branch2)
    
    # Branch 3: Large receptive field (global patterns)
    branch3 = layers.Conv1D(64, 15, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(1e-5), name='conv1d_branch3_1')(x)
    branch3 = layers.BatchNormalization(name='bn_branch3_1')(branch3)
    branch3 = layers.Conv1D(64, 15, padding='same', activation='relu',
                           kernel_regularizer=regularizers.l2(1e-5), name='conv1d_branch3_2')(branch3)
    branch3 = layers.BatchNormalization(name='bn_branch3_2')(branch3)
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate(axis=-1, name='multi_scale_concat')([branch1, branch2, branch3])
    
    # Residual block
    residual_input = layers.Conv1D(192, 1, padding='same', activation='relu', name='residual_projection')(x)
    residual = layers.Conv1D(192, 5, padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(1e-5), name='residual_conv1')(multi_scale)
    residual = layers.BatchNormalization(name='residual_bn1')(residual)
    residual = layers.Conv1D(192, 5, padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(1e-5), name='residual_conv2')(residual)
    residual = layers.BatchNormalization(name='residual_bn2')(residual)
    
    # Add residual connection
    x = layers.Add(name='residual_add')([residual_input, residual])
    x = layers.Activation('relu', name='residual_activation')(x)
    
    # Attention pooling instead of simple GlobalMaxPooling
    attention_weights = layers.Conv1D(1, 1, activation='softmax', name='attention_weights')(x)
    attended_features = layers.Multiply(name='attention_multiply')([x, attention_weights])
    
    # Combine different pooling strategies
    max_pooled = layers.GlobalMaxPooling1D(name='global_max_pool')(attended_features)
    avg_pooled = layers.GlobalAveragePooling1D(name='global_avg_pool')(attended_features)
    
    # Final feature vector
    features = layers.Concatenate(name='pooled_features')([max_pooled, avg_pooled])
    
    # Additional dense layer for feature transformation
    features = layers.Dense(256, activation='relu', 
                           kernel_regularizer=regularizers.l2(1e-4), name='feature_dense')(features)
    features = layers.BatchNormalization(name='feature_bn')(features)
    features = layers.Dropout(0.2, name='feature_dropout')(features)

    return keras.Model(inputs=input_layer, outputs=features, name="enhanced_cnn_network")
