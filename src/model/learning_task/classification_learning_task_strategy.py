""" Classification Learning Task Strategy """

import logging
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from src.model.learning_task.base_learning_task_strategy import BaseLearningTaskStrategy
from helper.enum.dataset.binary_threshold import BinaryThreshold

class ClassificationLearningTaskStrategy(BaseLearningTaskStrategy):
    """Strategy for classification learning tasks."""

    def get_loss_function(self):
        return tf.keras.losses.BinaryCrossentropy()

    def get_metrics(self):
        return [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Accuracy(name='accuracy')
        ]

    def process_targets(self, y):
        """Converts regression targets to binary classification targets."""
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        return (y >= BinaryThreshold.value.value).astype(int)

    def compile_model(self, model, learning_rate):
        """Compiles the model for classification."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.get_loss_function(),
            metrics=self.get_metrics()
        )
        logging.info("Classification model compiled with Adam optimizer.")

    def evaluate_model(self, y_true, y_pred, comet=None):
        """Evaluates the classification model and logs metrics."""
        y_true_b = self.process_targets(y_true)
        y_pred_b = (y_pred > 0.5).astype(int)

        metrics = {
            'f1_score': f1_score(y_true_b, y_pred_b),
            'accuracy': accuracy_score(y_true_b, y_pred_b),
            'precision': precision_score(y_true_b, y_pred_b),
            'recall': recall_score(y_true_b, y_pred_b),
            'roc_auc': roc_auc_score(y_true_b, y_pred)
        }

        logging.info(f"Evaluation metrics: {metrics}")
        if comet:
            comet.log_metrics(metrics)
