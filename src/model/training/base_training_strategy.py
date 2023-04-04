""" Base model strategy """
from abc import ABC, abstractmethod


class BaseModelStrategy(ABC):
    """ Base model strategy """

    @abstractmethod
    def train_and_evaluate_model(self, model_creation_strategy, dataset_tuple, batch_size, learning_rate, epoch):
        """ Train and evaluate model """
        pass
