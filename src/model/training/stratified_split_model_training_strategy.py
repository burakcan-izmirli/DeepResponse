""" Stratified split training strategy """
import logging
from tensorflow import keras

from src.model.training.base_training_strategy import BaseTrainingStrategy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, \
    accuracy_score, f1_score, matthews_corrcoef, auc


class StratifiedSplitTrainingStrategy(BaseTrainingStrategy):
    """ Random split training strategy """

    def train_and_evaluate_model(self, model_creation_strategy, dataset_iterator, batch_size, learning_rate, epoch):
        """ Train model and predict """
        for _ in dataset_iterator:
            dims, train_dataset, test_dataset, y_test = _
            model = model_creation_strategy.create_model(*dims, batch_size)
            # logging.info(model.summary())
            model.compile(loss=keras.losses.MeanSquaredError(),
                          optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                          metrics=[keras.metrics.MeanSquaredError(name='mse'),
                                   keras.metrics.RootMeanSquaredError(name='rmse'),
                                   keras.metrics.MeanAbsoluteError(name='mae'),
                                   r2_score])
            model.fit(train_dataset,
                      validation_data=test_dataset,
                      epochs=epoch,
                      verbose=1)