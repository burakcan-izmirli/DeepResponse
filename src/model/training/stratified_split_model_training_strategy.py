import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from src.model.training.base_training_strategy import BaseTrainingStrategy

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

            model = model_creation_strategy.create_model(
                drug_input_shape=drug_smiles_input_shape,
                cell_input_shape=cell_input_shape,
                selformer_trainable_layers=strategy_creator.selformer_trainable_layers
            )

            learning_task_strategy.compile_model(model, strategy_creator.learning_rate)

            checkpoint_path = self._get_checkpoint_path(strategy_creator, fold_idx)
            callbacks = self._get_callbacks(checkpoint_path, comet_logger)

            history = model.fit(
                train_dataset,
                epochs=strategy_creator.epoch,
                validation_data=valid_dataset,
                callbacks=callbacks,
                verbose=2
            )

            model.load_weights(checkpoint_path)
            y_pred = model.predict(test_dataset)
            fold_metrics = learning_task_strategy.evaluate_model(y_test_df, y_pred, comet_logger)
            all_fold_results.append(fold_metrics)

            current_val_loss = min(history.history['val_loss'])
            if current_val_loss < best_overall_val_loss:
                best_overall_val_loss = current_val_loss

        self._log_final_cv_results(all_fold_results, comet_logger)

    def _get_callbacks(self, checkpoint_path, comet_experiment):
        return [
            ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ]

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