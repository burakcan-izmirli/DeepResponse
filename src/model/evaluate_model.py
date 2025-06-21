""" Evaluate model """
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, \
    accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import Binarizer
from scipy.stats import pearsonr, spearmanr

from helper.enum.dataset.binary_threshold import BinaryThreshold


def binarize_data(data, threshold=BinaryThreshold.threshold.value):
    """
    Binarize dataset based on given threshold to calculate classification metrics
    :param data: dataset
    :param threshold: Threshold
    :return Binary dataset
    """
    if hasattr(data, 'values'):
        data = data.values
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return Binarizer(threshold=threshold).transform(data)


def evaluate_model(y_test, y_pred):
    """
    Evaluate model with different performance metrics
    :param y_test: y_test
    :param y_pred: y_pred
    :return: Evaluation metrics
    """

    np.save("y_test.npy", y_test)
    np.save("y_pred.npy", y_pred)

    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    y_test_binarized = binarize_data(y_test).flatten()
    y_pred_binarized = binarize_data(y_pred).flatten()

    return {"Mean Squared Error": mean_squared_error(y_test, y_pred),
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "Root Mean Squared Error": mean_squared_error(y_test, y_pred, squared=False),
            "R2 Score": r2_score(y_test, y_pred),
            "Pearson": pearsonr(y_test_flat, y_pred_flat),
            "Spearman": spearmanr(y_test_flat, y_pred_flat),
            "Accuracy": accuracy_score(y_test_binarized, y_pred_binarized),
            "Precision": precision_score(y_test_binarized, y_pred_binarized),
            "Recall": recall_score(y_test_binarized, y_pred_binarized),
            "F1 Score": f1_score(y_test_binarized, y_pred_binarized),
            "Matthew's Correlation Coef": matthews_corrcoef(y_test_binarized, y_pred_binarized)}
