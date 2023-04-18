""" Base training strategy """
from abc import ABC, abstractmethod


class BaseTrainingStrategy(ABC):
    """ Base training strategy """

    @abstractmethod
    def train_and_evaluate_model(self, model_creation_strategy, dataset_tuple, batch_size, learning_rate, epoch):
        """ Train and evaluate model """
        pass
