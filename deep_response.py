""" DeepResponse """
import logging
import os
from argparse import Namespace

from helper.argument_parser import argument_parser
from helper.seed_setter import set_seed
from src.strategy_creator import StrategyCreator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(asctime)s:%(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class DeepResponse:
    def __init__(self, args: Namespace):
        self.strategy_creator = StrategyCreator(args)

    def main(self):
        logging.info("DeepResponse started.")
        args = self.strategy_creator.args

        comet_logger = self.strategy_creator.get_comet_strategy().integrate_comet()

        logging.info(f"SELFormer Trainable Layers: {args.selformer_trainable_layers}")
        logging.info(f"Learning Task: {args.learning_task}, Split Type: {args.split_type}")
        logging.info(f"Learning Rate: {args.learning_rate}, Epochs: {args.epoch}, Batch Size: {args.batch_size}")

        set_seed(args.random_state)

        split_strategy_dict = self.strategy_creator.get_split_strategy()
        dataset_strategy = split_strategy_dict['dataset']
        model_training_strategy = split_strategy_dict['training']

        learning_task_strategy_instance = self.strategy_creator.get_learning_task_strategy()

        raw_dataset_dict = dataset_strategy.read_and_shuffle_dataset(args.random_state)

        dataset_input = dataset_strategy.prepare_dataset(
            raw_dataset_dict,
            args.split_type,
            args.batch_size,
            args.random_state,
            learning_task_strategy_instance
        )

        model_training_strategy.train_and_evaluate_model(
            self.strategy_creator,
            dataset_input,
            comet_logger
        )


if __name__ == '__main__':
    args = argument_parser()
    DeepResponse(args).main()
