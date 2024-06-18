""" Argument parser """
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from helper.enum.default_arguments import DefaultArguments


def argument_parser():
    """ Argument parser """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-uc", "--use_comet", default=False, type=bool, help="Whether to use comet or not")
    parser.add_argument("-ds", "--data_source", default='DEPMAP', type=str, help="['DEPMAP', 'gdsc', 'ccle', 'nci_60']")
    parser.add_argument("-es", "--evaluation_source", default=None, help="['DEPMAP', 'gdsc', 'ccle', 'nci_60']")
    parser.add_argument("-dt", "--data_type", default=DefaultArguments.data_type.value, type=str,
                        help="For all data sources: ['normal', 'l1000', 'l1000_cross_domain] "
                             "For just GDSC: ['pathway', 'pathway_reduced', 'digestive']")
    parser.add_argument("-st", "--split_type", default=DefaultArguments.split_type.value, type=str,
                        help="['random', 'cell_stratified', 'drug_stratified', 'drug_cell_stratified', 'cross_domain']")
    parser.add_argument("-rs", "--random_state", default=DefaultArguments.random_state.value, type=int,
                        help="Random State")
    parser.add_argument("-bs", "--batch_size", default=DefaultArguments.batch_size.value, type=int, help="Batch Size")
    parser.add_argument("-e", "--epoch", default=DefaultArguments.epoch.value, type=int, help="Epoch size")
    parser.add_argument("-lr", "--learning_rate", default=DefaultArguments.learning_rate.value, type=float,
                        help="Learning Rate")

    args = vars(parser.parse_args())

    # If evaluation source is not given then use data source as evaluation source.
    if args["evaluation_source"] is None:
        args["evaluation_source"] = args["data_source"]

    return args["use_comet"], args["data_source"], args["evaluation_source"], args["data_type"], args["split_type"], \
        args["random_state"], args["batch_size"], args["epoch"], args["learning_rate"]
