""" Argument parser """
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from helper.enum.default_arguments import DefaultArguments


def argument_parser():
    """ Argument parser """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--use_comet", default=False, type=bool, help="Whether to use comet or not")
    parser.add_argument("-d", "--data_type", default=DefaultArguments.data_type.value, type=str,
                        help="['normal', 'l1000', 'pathway', 'pathway_reduced', 'digestive']")
    parser.add_argument("-s", "--split_type", default=DefaultArguments.split_type.value, type=str,
                        help="['random', 'cell_stratified', 'drug_stratified', 'drug_cell_stratified']")
    parser.add_argument("-r", "--random_state", default=DefaultArguments.random_state.value, type=int,
                        help="Random State")
    parser.add_argument("-b", "--batch_size", default=DefaultArguments.batch_size.value, type=int, help="Batch Size")
    parser.add_argument("-e", "--epoch", default=DefaultArguments.epoch.value, type=int, help="Epoch size")
    parser.add_argument("-l", "--learning_rate", default=DefaultArguments.learning_rate.value, type=float,
                        help="Learning Rate")

    args = vars(parser.parse_args())

    return args["use_comet"], args["data_type"], args["split_type"], args["random_state"], args["batch_size"], \
        args["epoch"], args["learning_rate"]
