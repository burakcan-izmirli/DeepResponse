from strategies.learning_task.base_learning_task_strategy import BaseLearningTaskStrategy
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ClassificationLearningTaskStrategy(BaseLearningTaskStrategy):
    """ Classification learning task strategy """

    def get_loss_function(self):
        return keras.losses.BinaryCrossentropy(from_logits=True)

    def get_metrics(self):
        return [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]

    def compile_model(self, model, learning_rate):
        loss_function = self.get_loss_function()
        metrics = self.get_metrics()
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss_function, metrics=metrics)
        return model

    def evaluate_model(self, y_true, y_pred, comet=None):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
        if comet:
            comet.log_metrics(metrics)
        return metrics

    def visualize_results(self, y_true, y_pred, comet=None):
        y_prob = tf.sigmoid(y_pred).numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0,1], [0,1], linestyle='--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        plt.grid(True)
        if comet:
            comet.log_figure(figure_name='ROC Curve', figure=plt)
        plt.show()
