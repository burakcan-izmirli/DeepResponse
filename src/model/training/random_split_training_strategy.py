""" Random split training strategy """
import logging

from tensorflow import keras

from src.model.training.base_training_strategy import BaseTrainingStrategy
from src.model.evaluate_model import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, \
    accuracy_score, f1_score, matthews_corrcoef, auc


class RandomSplitTrainingStrategy(BaseTrainingStrategy):
    """ Random split training strategy """

    def train_and_evaluate_model(self, model_creation_strategy, dataset_tuple, batch_size, learning_rate, epoch):
        """ Train model and predict """
        dims, train_dataset, valid_dataset, test_dataset, y_test = dataset_tuple
        model = model_creation_strategy.create_model(*dims, batch_size)
        # logging.info(model.summary())
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[keras.metrics.MeanSquaredError(name='mse'),
                               keras.metrics.RootMeanSquaredError(name='rmse'),
                               keras.metrics.MeanAbsoluteError(name='mae'),
                               r2_score])
        model.fit(train_dataset,
                  validation_data=valid_dataset,
                  epochs=epoch,
                  verbose=2)

        predictions = model.predict(test_dataset, verbose=2)
        logging.info(evaluate_model(y_test, predictions))