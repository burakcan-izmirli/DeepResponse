""" Random split training strategy """
import logging
from tensorflow import keras
from sklearn.metrics import r2_score

from src.model.training.base_training_strategy import BaseTrainingStrategy
from src.model.evaluate_model import evaluate_model
from src.model.visualize_results import visualize_results

def r2_loss(y_true, y_pred):
    SS_res =  keras.backend.sum(keras.backend.square(y_true - y_pred))
    SS_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return 1 - ( 1 - SS_res/(SS_tot + keras.backend.epsilon()))

class RandomSplitTrainingStrategy(BaseTrainingStrategy):
    """ Random split training strategy """

    def train_and_evaluate_model(self, model_creation_strategy, dataset_tuple, batch_size, learning_rate, epoch):
        """ Train model and predict """
        dims, train_dataset, valid_dataset, test_dataset, y_test = dataset_tuple
        model = model_creation_strategy.create_model(*dims, batch_size)
        # logging.info(model.summary())
        model.compile(loss=r2_score,
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
        visualize_results(y_test.values, predictions)
        logging.info(evaluate_model(y_test.values, predictions))
