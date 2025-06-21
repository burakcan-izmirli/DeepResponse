import os
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from src.model.training.base_training_strategy import BaseTrainingStrategy

class RandomSplitTrainingStrategy(BaseTrainingStrategy):
    def train_and_evaluate_model(self, strategy_creator, dataset_input, comet_logger):
        """
        Train and evaluate model using random split strategy.
        
        Args:
            strategy_creator: Strategy creator instance
            dataset_input: Tuple containing (dims, train_dataset, valid_dataset, test_dataset, y_test_df)
            comet_logger: Comet ML logger instance
        """
        try:
            learning_task_strategy = strategy_creator.get_learning_task_strategy()
            model_creation_strategy = strategy_creator.get_model_creation_strategy()

            dims, train_dataset, valid_dataset, test_dataset, y_test_df = dataset_input
            drug_smiles_input_shape, cell_input_shape = dims

            logging.info("Creating model...")
            model = model_creation_strategy.create_model(
                drug_input_shape=drug_smiles_input_shape,
                cell_input_shape=cell_input_shape,
                selformer_trainable_layers=strategy_creator.selformer_trainable_layers
            )

            learning_task_strategy.compile_model(model, strategy_creator.learning_rate)

            checkpoint_path = self._get_checkpoint_path(strategy_creator)
            callbacks = self._get_callbacks(checkpoint_path, comet_logger)

            logging.info(f"Starting training for {strategy_creator.epoch} epochs...")
            model.fit(
                train_dataset,
                epochs=strategy_creator.epoch,
                validation_data=valid_dataset,
                callbacks=callbacks,
                verbose=2
            )

            # Verify checkpoint exists before loading
            if not os.path.exists(checkpoint_path):
                logging.warning(f"Checkpoint not found at {checkpoint_path}. Using current model weights.")
            else:
                logging.info("Loading best model for evaluation...")
                model.load_weights(checkpoint_path)

            logging.info("Evaluating model on the test set...")
            y_pred = model.predict(test_dataset, verbose=0)
            learning_task_strategy.evaluate_model(y_test_df, y_pred, comet_logger)
            
            logging.info("Training and evaluation completed successfully.")
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
        finally:
            # Clean up TensorFlow session
            tf.keras.backend.clear_session()

    def _get_callbacks(self, checkpoint_path, comet_experiment):
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10)
        ]
        if comet_experiment:
            callbacks.append(self.CometMetricsCallback(comet_experiment))
        return callbacks

    def _get_checkpoint_path(self, strategy_creator):
        prefix = "selformer_cnn"
        if strategy_creator.selformer_trainable_layers == 0:
            prefix += "_frozen"
        elif strategy_creator.selformer_trainable_layers > 0:
            prefix += f"_stl{strategy_creator.selformer_trainable_layers}"

        checkpoint_dir = f'./checkpoints_{prefix}_random'
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, 'model_best.keras')

    class CometMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, comet_experiment):
            self.comet_experiment = comet_experiment

        def on_epoch_end(self, epoch, logs=None):
            if self.comet_experiment and logs:
                self.comet_experiment.log_metrics(logs, epoch=epoch)