""" Base training strategy """
from abc import ABC, abstractmethod

class BaseTrainingStrategy(ABC):
    @abstractmethod
    def train_and_evaluate_model(self, strategy_creator, dataset_input, comet_logger):
        """ Train and evaluate model """
        pass