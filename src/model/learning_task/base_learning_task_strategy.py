from abc import ABC, abstractmethod

class BaseLearningTaskStrategy(ABC):
    """Base class for learning task-specific strategies."""

    @abstractmethod
    def get_loss_function(self):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def evaluate_model(self, y_true, y_pred, comet=None):
        pass

    @abstractmethod
    def visualize_results(self, y_true, y_pred, comet=None):
        pass
