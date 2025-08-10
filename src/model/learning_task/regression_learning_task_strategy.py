""" Regression Learning Task Strategy """
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from src.model.evaluate_model import evaluate_model
from src.model.visualize_results import visualize_results
from src.model.learning_task.base_learning_task_strategy import BaseLearningTaskStrategy
from sklearn.metrics import r2_score


class SafeR2Score(tf.keras.metrics.Metric):
    """
    Custom R² score metric that handles edge cases gracefully.
    Prevents NaN values by checking variance prerequisites.
    """
    
    def __init__(self, name='r2_score', **kwargs):
        super(SafeR2Score, self).__init__(name=name, **kwargs)
        self.total_sum_squares = self.add_weight(name='total_sum_squares', initializer='zeros')
        self.residual_sum_squares = self.add_weight(name='residual_sum_squares', initializer='zeros')
        self.sum_y_true = self.add_weight(name='sum_y_true', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state with new batch of data."""
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            y_true = tf.multiply(y_true, sample_weight)
            y_pred = tf.multiply(y_pred, sample_weight)

        # Update counts and sums
        batch_count = tf.cast(tf.size(y_true), self._dtype)
        self.count.assign_add(batch_count)
        self.sum_y_true.assign_add(tf.reduce_sum(y_true))
        
        # Calculate mean on the fly to avoid numerical issues
        y_mean = self.sum_y_true / tf.maximum(self.count, 1e-8)
        
        # Update sum of squares
        self.total_sum_squares.assign_add(tf.reduce_sum(tf.square(y_true - y_mean)))
        self.residual_sum_squares.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))

    def result(self):
        """Calculate R² score with safety checks."""
        
        # Check if we have enough samples and variance
        safe_tss = tf.maximum(self.total_sum_squares, 1e-8)
        
        # Calculate R²
        r2 = 1.0 - (self.residual_sum_squares / safe_tss)
        
        # Check for undefined conditions
        is_undefined = tf.logical_or(
            tf.less(self.count, 2.0),  # Less than 2 samples
            tf.less(self.total_sum_squares, 1e-8)  # No variance in targets
        )
        
        # Return -999 as sentinel value for undefined cases, otherwise R²
        return tf.where(is_undefined, -999.0, r2)

    def reset_state(self):
        """Reset all metric states."""
        self.total_sum_squares.assign(0.0)
        self.residual_sum_squares.assign(0.0)
        self.sum_y_true.assign(0.0)
        self.count.assign(0.0)


class RegressionLearningTaskStrategy(BaseLearningTaskStrategy):
    """ Strategy for regression learning tasks. """

    def get_loss_function(self):
        return tf.keras.losses.Huber()

    def get_metrics(self):
        """Get metrics for regression with safe R² calculation."""
        return [
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            SafeR2Score(name='r2_score')
        ]

    def process_targets(self, y):
        """ Ensures targets are in the correct format for regression. """
        return y

    def compile_model(self, model, learning_rate):
        """ 
        Compiles the model for regression with proper error handling.
        
        Args:
            model: TensorFlow/Keras model to compile
            learning_rate: Learning rate for the optimizer
        """
        try:
            if learning_rate <= 0:
                raise ValueError(f"Learning rate must be positive, got: {learning_rate}")

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=self.get_loss_function(),
                metrics=self.get_metrics()
            )
            logging.info(f"Regression model compiled successfully with Adam optimizer (lr={learning_rate}).")
        except Exception as e:
            logging.error(f"Failed to compile regression model: {e}")
            raise

    def evaluate_model(self, y_true, y_pred, comet=None, experiment_context=None):
        """ Evaluates the regression model and logs metrics. """
        if isinstance(y_true, (pd.DataFrame, pd.Series)):
            y_true = y_true.to_numpy()

        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        metrics = evaluate_model(y_true, y_pred)

        logging.info(f"Evaluation metrics: {metrics}")
        if comet:
            comet.log_metrics({
                "Mean Squared Error": metrics["Mean Squared Error"],
                "Mean Absolute Error": metrics["Mean Absolute Error"],
                "Root Mean Squared Error": metrics["Root Mean Squared Error"],
                "R2 Score": metrics["R2 Score"],
                "Pearson Correlation": metrics["Pearson"][0],
                "Pearson p-value": metrics["Pearson"][1],
                "Spearman Correlation": metrics["Spearman"].correlation,
                "Spearman p-value": metrics["Spearman"].pvalue,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "Matthew's Correlation Coef": metrics["Matthew's Correlation Coef"]
            })

        # Generate dynamic filename prefix based on experiment context
        filename_prefix = ""
        if experiment_context:
            split_type = experiment_context.get('split_type', 'unknown')
            trainable_layers = experiment_context.get('selformer_trainable_layers', 'unknown')
            data_source = experiment_context.get('data_source', 'unknown')
            fold_idx = experiment_context.get('fold_idx', '')
            
            filename_prefix = f"{data_source}_{split_type}_stl{trainable_layers}"
            if fold_idx != '':
                filename_prefix += f"_fold{fold_idx}"
            filename_prefix += "_"

        visualize_results(y_true, y_pred, comet, filename_prefix)
        
        return metrics
