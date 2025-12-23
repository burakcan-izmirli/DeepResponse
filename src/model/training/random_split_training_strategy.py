import os
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from src.model.training.base_training_strategy import BaseTrainingStrategy
from src.model.training.advanced_schedulers import get_advanced_callbacks, get_scheduler_recommendation

class GradientNormLogger(Callback):
    def __init__(self, comet_experiment=None):
        super().__init__()
        self.comet = comet_experiment
    def on_train_batch_end(self, batch, logs=None):
        # Not easily accessible without custom training loop; skip
        pass
    def on_epoch_end(self, epoch, logs=None):
        # Placeholder: gradient norms require custom train_step override; skip to avoid errors
        if self.comet:
            try:
                self.comet.log_metric('epoch', epoch, step=epoch)
            except Exception:
                pass

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

            # Optionally cache datasets for speed
            if strategy_creator.args.cache_datasets:
                train_dataset = train_dataset.cache()
                valid_dataset = valid_dataset.cache()
                test_dataset = test_dataset.cache()

            # Log scheduler recommendation
            scheduler_rec = get_scheduler_recommendation(
                model_type='hybrid',
                dataset_size='large',
                selformer_layers=strategy_creator.selformer_trainable_layers
            )
            logging.info(f"Using {scheduler_rec['type']} scheduler: {scheduler_rec['reason']}")

            logging.info("Creating model...")
            model = model_creation_strategy.create_model(
                drug_input_shape=drug_smiles_input_shape,
                cell_input_shape=cell_input_shape,
                selformer_trainable_layers=strategy_creator.selformer_trainable_layers
            )

            learning_task_strategy.compile_model(model, strategy_creator.learning_rate)

            checkpoint_path = self._get_checkpoint_path(strategy_creator)
            
            # Calculate steps per epoch for advanced schedulers
            steps_per_epoch = self._calculate_steps_per_epoch(train_dataset, strategy_creator.batch_size)
            
            # Use advanced callbacks instead of basic ones
            callbacks = get_advanced_callbacks(
                strategy_creator, 
                checkpoint_path, 
                steps_per_epoch, 
                comet_logger
            )
            callbacks.append(GradientNormLogger(comet_logger))

            model.fit(
                train_dataset,
                epochs=strategy_creator.epoch,
                validation_data=valid_dataset,
                callbacks=callbacks,
                verbose=2
            )

            # Load best model weights
            best_model_path = checkpoint_path.replace('.keras', '_best_val.h5')
            if os.path.exists(best_model_path):
                model.load_weights(best_model_path)
                logging.info(f"Loaded best model from {best_model_path}")
            elif os.path.exists(checkpoint_path):
                logging.info("Loading best model for evaluation...")
                model.load_weights(checkpoint_path)
            else:
                logging.warning(f"No checkpoint found. Using current model weights.")

            logging.info("Evaluating model on the test set...")
            y_pred = model.predict(test_dataset, verbose=0)
            
            # Create experiment context for dynamic filenames
            experiment_context = {
                'split_type': strategy_creator.split_type,
                'selformer_trainable_layers': strategy_creator.selformer_trainable_layers,
                'data_source': strategy_creator.data_source
            }
            
            learning_task_strategy.evaluate_model(y_test_df, y_pred, comet_logger, experiment_context)
            
            logging.info("Training and evaluation completed successfully.")
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
        finally:
            # Clean up TensorFlow session
            tf.keras.backend.clear_session()

    def _calculate_steps_per_epoch(self, train_dataset, batch_size):
        """Calculate steps per epoch for the training dataset."""
        try:
            # Try to get dataset size from cardinality
            dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
            if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
                # Fallback: estimate based on iteration
                steps = 0
                for _ in train_dataset.take(100):  # Sample to estimate
                    steps += 1
                return max(steps * 10, 1000)  # Rough estimate
            return max(dataset_size, 1000)
        except:
            # Fallback for any errors
            return 1000

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