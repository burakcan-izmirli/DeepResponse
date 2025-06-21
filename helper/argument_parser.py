""" Argument parser """
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from helper.enum.default_arguments import DefaultArguments


def argument_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-uc", "--use_comet", default=DefaultArguments.comet.value,
                        type=lambda x: (str(x).lower() == 'true'), help="Whether to use comet or not")
    parser.add_argument('-lt', "--learning_task", default=DefaultArguments.learning_task.value,
                        type=str, help="['classification', 'regression']")
    parser.add_argument("-ds", "--data_source", default=DefaultArguments.data_source.value,
                        type=str, help="['depmap', 'gdsc', 'ccle', 'nci_60']")
    parser.add_argument("-es", "--evaluation_source", default=None,
                        type=str, help="['depmap', 'gdsc', 'ccle', 'nci_60']")
    parser.add_argument("-dt", "--data_type", default=DefaultArguments.data_type.value, type=str,
                        help="For all data sources: ['normal', 'l1000', 'l1000_cross_domain'] "
                             "For just GDSC: ['pathway', 'pathway_reduced', 'digestive']")
    parser.add_argument("-st", "--split_type", default=DefaultArguments.split_type.value, type=str,
                        help="['random', 'cell_stratified', 'drug_stratified', 'drug_cell_stratified', 'cross_domain']")
    parser.add_argument("-rs", "--random_state", default=DefaultArguments.random_state.value,
                        type=int, help="Random state for reproducibility")
    parser.add_argument("-bs", "--batch_size", default=DefaultArguments.batch_size.value,
                        type=int, help="Batch size for training")
    parser.add_argument("-e", "--epoch", default=DefaultArguments.epoch.value,
                        type=int, help="Number of epochs for training")
    parser.add_argument("-lr", "--learning_rate", default=DefaultArguments.learning_rate.value,
                        type=float, help="Learning rate for optimizer")
    parser.add_argument("-stl", "--selformer_trainable_layers",
                        default=DefaultArguments.selformer_trainable_layers.value,
                        type=int, help="Number of trainable layers in SELFormer")

    args = parser.parse_args()
    return args
