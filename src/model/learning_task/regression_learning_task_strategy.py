""" Regression Learning Task Strategy """
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from src.model.evaluate_model import evaluate_model
from src.model.visualize_results import visualize_results
from src.model.learning_task.base_learning_task_strategy import BaseLearningTaskStrategy
from sklearn.metrics import r2_score


class RegressionLearningTaskStrategy(BaseLearningTaskStrategy):
    """ Strategy for regression learning tasks. """

    def get_loss_function(self):
        return tf.keras.losses.Huber()

    def get_metrics(self):
        def r2(y_true, y_pred):
            return tf.py_function(r2_score, (y_true, y_pred), tf.double)
        r2.__name__ = 'r2_score'
        return [
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            r2
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

    def evaluate_model(self, y_true, y_pred, comet=None):
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

        visualize_results(y_true, y_pred, comet)
