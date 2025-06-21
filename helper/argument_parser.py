""" 
Command-line argument parser for DeepResponse.

This module handles all command-line arguments for the DeepResponse system,
providing validation and default values for training parameters.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from helper.enum.default_arguments import DefaultArguments


def argument_parser():
    """
    Parse command-line arguments for DeepResponse.
    
    Returns:
        Namespace: Parsed arguments with validation
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response",
        epilog="For more information, visit: https://github.com/burakcan-izmirli/DeepResponse"
    )

    # Logging and monitoring
    parser.add_argument("-uc", "--use_comet", default=DefaultArguments.comet.value,
                        type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to use Comet ML for experiment tracking")

    # Task configuration
    parser.add_argument('-lt', "--learning_task", default=DefaultArguments.learning_task.value,
                        type=str, choices=['classification', 'regression'],
                        help="Learning task type")
    
    # Dataset configuration
    parser.add_argument("-ds", "--data_source", default=DefaultArguments.data_source.value,
                        type=str, choices=['depmap', 'gdsc', 'ccle', 'nci_60'],
                        help="Primary data source for training")
    parser.add_argument("-es", "--evaluation_source", default=None,
                        type=str, choices=['depmap', 'gdsc', 'ccle', 'nci_60'],
                        help="Evaluation data source for cross-domain validation")
    parser.add_argument("-dt", "--data_type", default=DefaultArguments.data_type.value, type=str,
                        choices=['normal', 'l1000', 'l1000_cross_domain', 'pathway', 'pathway_reduced', 'digestive'],
                        help="Data type configuration")
    parser.add_argument("-st", "--split_type", default=DefaultArguments.split_type.value, type=str,
                        choices=['random', 'cell_stratified', 'drug_stratified', 'drug_cell_stratified', 'cross_domain'],
                        help="Dataset splitting strategy")
    
    # Training hyperparameters
    parser.add_argument("-rs", "--random_state", default=DefaultArguments.random_state.value,
                        type=int, help="Random seed for reproducibility (>= 0)")
    parser.add_argument("-bs", "--batch_size", default=DefaultArguments.batch_size.value,
                        type=int, help="Training batch size (> 0)")
    parser.add_argument("-e", "--epoch", default=DefaultArguments.epoch.value,
                        type=int, help="Number of training epochs (> 0)")
    parser.add_argument("-lr", "--learning_rate", default=DefaultArguments.learning_rate.value,
                        type=float, help="Optimizer learning rate (> 0)")
    
    # Model configuration
    parser.add_argument("-stl", "--selformer_trainable_layers",
                        default=DefaultArguments.selformer_trainable_layers.value,
                        type=int, help="Number of trainable layers in SELFormer (-1 for all, 0 for frozen, >0 for specific count)")

    args = parser.parse_args()
    _validate_parsed_args(args)
    return args


def _validate_parsed_args(args):
    """
    Validate parsed arguments for logical consistency.
    
    Args:
        args: Parsed arguments namespace
        
    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    # Validate numerical parameters
    if args.learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got: {args.learning_rate}")
    
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got: {args.batch_size}")
    
    if args.epoch <= 0:
        raise ValueError(f"Number of epochs must be positive, got: {args.epoch}")
    
    if args.random_state < 0:
        raise ValueError(f"Random state must be non-negative, got: {args.random_state}")
    
    # Validate cross-domain configuration
    if args.split_type == 'cross_domain' and args.evaluation_source is None:
        raise ValueError("evaluation_source is required when split_type is 'cross_domain'")
    
    # Validate data source compatibility
    if args.evaluation_source == args.data_source and args.split_type == 'cross_domain':
        raise ValueError("evaluation_source must be different from data_source for cross_domain split")
    
    print(f"✓ Argument validation passed: {args.learning_task} task on {args.data_source} data")
