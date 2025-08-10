import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from src.model.training.base_training_strategy import BaseTrainingStrategy
from src.model.training.advanced_schedulers import get_advanced_callbacks, get_scheduler_recommendation

class StratifiedSplitTrainingStrategy(BaseTrainingStrategy):
    def train_and_evaluate_model(self, strategy_creator, dataset_iterator, comet_logger):
        learning_task_strategy = strategy_creator.get_learning_task_strategy()
        model_creation_strategy = strategy_creator.get_model_creation_strategy()

        all_fold_results = []
        best_overall_val_loss = float('inf')

        for fold_idx, data_fold in enumerate(dataset_iterator):
            logging.info(f"\n----- Starting CV Fold {fold_idx + 1} -----")
            tf.keras.backend.clear_session()

            dims, train_dataset, valid_dataset, test_dataset, y_test_df = data_fold
            drug_smiles_input_shape, cell_input_shape = dims

            # Log scheduler recommendation
            scheduler_rec = get_scheduler_recommendation(
                model_type='hybrid',
                dataset_size='large',
                selformer_layers=strategy_creator.selformer_trainable_layers
            )
            logging.info(f"Using {scheduler_rec['type']} scheduler: {scheduler_rec['reason']}")

            model = model_creation_strategy.create_model(
                drug_input_shape=drug_smiles_input_shape,
                cell_input_shape=cell_input_shape,
                selformer_trainable_layers=strategy_creator.selformer_trainable_layers
            )

            learning_task_strategy.compile_model(model, strategy_creator.learning_rate)

            checkpoint_path = self._get_checkpoint_path(strategy_creator, fold_idx)
            
            # Calculate steps per epoch for advanced schedulers
            steps_per_epoch = self._calculate_steps_per_epoch(train_dataset, strategy_creator.batch_size)
            
            # Use advanced callbacks instead of basic ones
            callbacks = get_advanced_callbacks(
                strategy_creator, 
                checkpoint_path, 
                steps_per_epoch, 
                comet_logger
            )

            logging.info(f"Training with {len(callbacks)} advanced callbacks including {scheduler_rec['type']} scheduler")

            history = model.fit(
                train_dataset,
                epochs=strategy_creator.epoch,
                validation_data=valid_dataset,
                callbacks=callbacks,
                verbose=2
            )

            # Load best model weights
            best_model_path = checkpoint_path.replace('.h5', '_best_val.h5')
            if os.path.exists(best_model_path):
                model.load_weights(best_model_path)
                logging.info(f"Loaded best model from {best_model_path}")
            else:
                model.load_weights(checkpoint_path)

            y_pred = model.predict(test_dataset, verbose=0)
            
            # Create experiment context for dynamic filenames
            experiment_context = {
                'split_type': strategy_creator.split_type,
                'selformer_trainable_layers': strategy_creator.selformer_trainable_layers,
                'data_source': strategy_creator.data_source,
                'fold_idx': fold_idx + 1
            }
            
            fold_metrics = learning_task_strategy.evaluate_model(y_test_df, y_pred, comet_logger, experiment_context)
            all_fold_results.append(fold_metrics)

            current_val_loss = min(history.history['val_loss'])
            if current_val_loss < best_overall_val_loss:
                best_overall_val_loss = current_val_loss

        self._log_final_cv_results(all_fold_results, comet_logger)

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

    def _get_checkpoint_path(self, strategy_creator, fold_idx):
        prefix = "selformer_cnn"
        if strategy_creator.selformer_trainable_layers == 0:
            prefix += "_frozen"
        elif strategy_creator.selformer_trainable_layers > 0:
            prefix += f"_stl{strategy_creator.selformer_trainable_layers}"

        checkpoint_dir = f'./checkpoints_{prefix}_{strategy_creator.split_type}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f'model_best_fold_{fold_idx + 1}.h5')

    def _log_final_cv_results(self, all_fold_results, comet_logger):
        if not all_fold_results:
            return

        avg_metrics = {key: np.mean([res[key] for res in all_fold_results]) for key in all_fold_results[0] if isinstance(all_fold_results[0][key], (int, float))}
        logging.info(f"Average CV Metrics: {avg_metrics}")
        if comet_logger:
            comet_logger.log_metrics({f"avg_{k}": v for k, v in avg_metrics.items()})