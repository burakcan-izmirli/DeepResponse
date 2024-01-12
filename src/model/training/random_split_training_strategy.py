""" Random split training strategy """
import logging
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score

from src.model.training.base_training_strategy import BaseTrainingStrategy
from src.model.evaluate_model import evaluate_model
from src.model.visualize_results import visualize_results


def r2_score(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))


class RandomSplitTrainingStrategy(BaseTrainingStrategy):
    """ Random split training strategy """

    def train_and_evaluate_model(self, model_creation_strategy, dataset_tuple, batch_size, learning_rate, epoch, comet):
        """ Train model and predict """
        dims, train_dataset, valid_dataset, test_dataset, y_test = dataset_tuple
        model = model_creation_strategy.create_model(*dims, batch_size)
        # logging.info(model.summary())
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                  decay_steps=100000,
                                                                  decay_rate=0.95)

        model.compile(loss=keras.losses.Huber(),
                      optimizer=keras.optimizers.Adam(lr_schedule),
                      metrics=[keras.metrics.MeanSquaredError(name='mse'),
                               keras.metrics.RootMeanSquaredError(name='rmse'),
                               keras.metrics.MeanAbsoluteError(name='mae'),
                               r2_score])
        model.fit(train_dataset,
                  validation_data=valid_dataset,
                  epochs=epoch,
                  verbose=2)

        predictions = model.predict(test_dataset, verbose=2)
        visualize_results(y_test.values, predictions, comet)
        logging.info(evaluate_model(y_test.values, predictions))

# %%
