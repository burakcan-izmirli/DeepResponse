import logging
from tensorflow import keras
from sklearn.metrics import r2_score as sklearn_r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model

from src.model.training.base_training_strategy import BaseTrainingStrategy
from src.model.evaluate_model import evaluate_model
from src.model.visualize_results import visualize_results

from src.model.build.graph_neural_network.edge_network import EdgeNetwork
from src.model.build.graph_neural_network.message_passing import MessagePassing
from src.model.build.graph_neural_network.partition_padding import PartitionPadding
from src.model.build.graph_neural_network.transformer_encoder import TransformerEncoder


# Define custom R2 score metric for TensorFlow/Keras
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


class RandomSplitTrainingStrategy(BaseTrainingStrategy):
    """Random split training strategy"""

    def train_and_evaluate_model(self, model_creation_strategy, dataset_tuple, batch_size, learning_rate, epoch, comet, learning_task_strategy):
        """ Train model and predict """
        dims, train_dataset, valid_dataset, test_dataset, y_test = dataset_tuple
        
        # Path where the model is saved
        checkpoint_path = 'model_checkpoint.h5'

        # # Check if the model exists and load it
        # try:
        #     # Pass the custom `MessagePassing`, `TransformerEncoder`, `MPNN`, and other layers in `custom_objects`
        #     model = load_model(checkpoint_path, 
        #                        custom_objects={'r2_score': r2_score, 
        #                                        'EdgeNetwork': EdgeNetwork,
        #                                        'MessagePassing': MessagePassing,
        #                                        'PartitionPadding': PartitionPadding, 
        #                                        'TransformerEncoder': TransformerEncoder})
        #     print("Model loaded successfully from checkpoint.")
        # except Exception as e:
        #     print(f"Error loading model: {e}. Creating a new model.")
        #     # If the model does not exist, create a new one
        #     model = model_creation_strategy.create_model(*dims, batch_size)
       
        # Create a new model using the model creation strategy
        model = model_creation_strategy.create_model(*dims, batch_size)    
        
        # Compile the model using the task-specific strategy
        model = learning_task_strategy.compile_model(model, learning_rate)

        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

        # Create a ModelCheckpoint callback to save the model
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,  # Path where the model is saved
            monitor='val_loss',        # Monitor validation loss
            save_best_only=True,       # Save only the best model
            save_weights_only=False,   # Save the entire model (not just weights)
            verbose=1
        )

        # Comet Integration
        if comet:
            comet_callback = comet.get_callback('keras')  # Get the Keras callback for Comet
            
            # Fit the model with ReduceLROnPlateau, ModelCheckpoint, and Comet callbacks
            model.fit(train_dataset,
                      validation_data=valid_dataset,
                      epochs=epoch,
                      callbacks=[comet_callback, reduce_lr, checkpoint],  # Add comet callback, ReduceLROnPlateau, and ModelCheckpoint
                      verbose=2)
        else:
            # Fit the model with ReduceLROnPlateau and ModelCheckpoint
            model.fit(train_dataset,
                      validation_data=valid_dataset,
                      epochs=epoch,
                      callbacks=[reduce_lr, checkpoint],  # Add ReduceLROnPlateau and ModelCheckpoint callbacks
                      verbose=2)

        # Make predictions on the test dataset
        predictions = model.predict(test_dataset, verbose=2)
        
        # Explicitly log the R2 score for the test set using sklearn
        test_r2 = sklearn_r2_score(y_test.values, predictions)
        if comet:
            comet.log_metric("r2_score_test", test_r2)

        # Visualize the results
        learning_task_strategy.visualize_results(y_test.values, predictions, comet)
        
        # Log evaluation metrics
        logging.info(learning_task_strategy.evaluate_model(y_test.values, predictions))
