from abc import ABC, abstractmethod

class BaseLearningTaskStrategy(ABC):
    """Base strategy for learning tasks."""

    @abstractmethod
    def get_loss_function(self):
        """Returns the loss function for the task."""
        ...

    @abstractmethod
    def get_metrics(self):
        """Returns a list of metrics for the task."""
        ...

    @abstractmethod
    def process_targets(self, y):
        """Processes target values for the specific task."""
        ...

    @abstractmethod
    def compile_model(self, model, learning_rate):
        """Compiles the model with task-specific settings."""
        ...

    @abstractmethod
    def evaluate_model(self, y_true, y_pred, comet=None):
        """Evaluates the model and logs metrics."""
        ...
