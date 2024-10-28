""" Main file of deep response """
from comet_ml import Experiment
import logging
import tensorflow as tf

from helper.argument_parser import argument_parser
from helper.seed_setter import set_seed
from src.strategy_creator import StrategyCreator

tf.config.run_functions_eagerly(True)


class DeepResponse(StrategyCreator):
    """ DeepResponse"""

    def main(self):
        """ Main function """
        logging.info("DeepResponse was started.")

        comet = self.get_comet_strategy().integrate_comet()

        set_seed(self.random_state)

        split_strategy = self.get_split_strategy()
        dataset_strategy = split_strategy['dataset']
        model_training_strategy = split_strategy['training']

        learning_task_strategy = self.get_learning_task_strategy()
        raw_dataset = dataset_strategy.read_and_shuffle_dataset(self.random_state)
        dataset_iterator = dataset_strategy.prepare_dataset(
            raw_dataset, self.split_type, self.batch_size, self.random_state, learning_task_strategy
        )

        model_creation_strategy = self.get_model_creation_strategy()

        model_training_strategy.train_and_evaluate_model(
            model_creation_strategy, dataset_iterator, self.batch_size, 
            self.learning_rate, self.epoch, comet, learning_task_strategy
        )


if __name__ == '__main__':
    DeepResponse(*argument_parser()).main()
