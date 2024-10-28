""" Regression Learning Task Strategy """
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from src.model.learning_task.base_learning_task_strategy import BaseLearningTaskStrategy

class RegressionLearningTaskStrategy(BaseLearningTaskStrategy):
    """ Strategy for regression learning tasks. """

    def get_loss_function(self):
        return keras.losses.Huber()

    def get_metrics(self):
        return [
            keras.metrics.MeanSquaredError(name='mse'),
            keras.metrics.RootMeanSquaredError(name='rmse'),
            keras.metrics.MeanAbsoluteError(name='mae'),
            self.r2_score_tf
        ]

    def process_targets(self, y):
        """ Pass-through for regression targets, returns pic50 values unchanged. """
        return y['pic50']

    def compile_model(self, model, learning_rate):
        loss_function = self.get_loss_function()
        metrics = self.get_metrics()
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss_function, metrics=metrics)
        return model

    def r2_score_tf(self, y_true, y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res / (ss_tot + keras.backend.epsilon())

    def evaluate_model(self, y_true, y_pred, comet=None):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics = {'mse': mse, 'mae': mae, 'r2': r2}
        if comet:
            comet.log_metrics(metrics)
        return metrics

    def visualize_results(self, y_true, y_pred, comet=None):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.grid(True)
        if comet:
            comet.log_figure(figure_name='Regression Results', figure=plt)
        plt.show()
