""" 
DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response 
with Deep Learning Based Pharmacogenomic Modelling

This module provides the main entry point for the DeepResponse system,
which employs multi-omics profiles and drug molecular features to predict
cancer cell drug sensitivity using hybrid convolutional and graph-transformer
deep neural networks.
"""
import logging
import os
import time
from argparse import Namespace
from contextlib import contextmanager

from helper.argument_parser import argument_parser
from helper.seed_setter import set_seed
from src.strategy_creator import StrategyCreator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(asctime)s:%(message)s')
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class DeepResponse:
    def __init__(self, args: Namespace):
        self.strategy_creator = StrategyCreator(args)

    @contextmanager
    def _execution_timer(self):
        """Context manager to track execution time."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    def main(self):
        """
        Main execution pipeline for DeepResponse.
        
        Raises:
            SystemExit: On any critical error during execution
        """
        with self._execution_timer():
            try:
                logging.info("DeepResponse started.")
                args = self.strategy_creator.args

                # Initialize Comet logging
                comet_logger = self._initialize_comet_logger()

                # Log configuration
                self._log_configuration(args)

                # Set random seed for reproducibility
                set_seed(args.random_state)

                # Initialize strategies
                strategies = self._initialize_strategies()
                
                # Load and prepare dataset
                dataset_input = self._prepare_dataset(strategies, args)
                
                # Train and evaluate model
                self._train_and_evaluate(strategies, dataset_input, comet_logger)
                
                logging.info("DeepResponse completed successfully.")
                
            except ValueError as e:
                logging.error(f"Configuration error: {e}")
                raise SystemExit(1) from e
            except FileNotFoundError as e:
                logging.error(f"Dataset file not found: {e}")
                raise SystemExit(1) from e
            except Exception as e:
                logging.error(f"Unexpected error during execution: {e}")
                raise SystemExit(1) from e

    def _initialize_comet_logger(self):
        """Initialize Comet ML logging strategy."""
        return self.strategy_creator.get_comet_strategy().integrate_comet()

    def _log_configuration(self, args):
        """Log current configuration parameters."""
        config_summary = f"""
        ================== DeepResponse Configuration ==================
        Data Source: {args.data_source}
        Evaluation Source: {args.evaluation_source or 'None'}
        Data Type: {args.data_type}
        Split Type: {args.split_type}
        Learning Task: {args.learning_task}
        
        Training Parameters:
        - Learning Rate: {args.learning_rate}
        - Epochs: {args.epoch}
        - Batch Size: {args.batch_size}
        - Random State: {args.random_state}
        
        Model Parameters:
        - SELFormer Trainable Layers: {args.selformer_trainable_layers}
        
        Logging:
        - Use Comet: {args.use_comet}
        ============================================================
        """
        logging.info(config_summary)

    def _initialize_strategies(self):
        """Initialize all required strategies."""
        split_strategy_dict = self.strategy_creator.get_split_strategy()
        learning_task_strategy = self.strategy_creator.get_learning_task_strategy()
        
        return {
            'dataset': split_strategy_dict['dataset'],
            'training': split_strategy_dict['training'],
            'learning_task': learning_task_strategy
        }

    def _prepare_dataset(self, strategies, args):
        """Load and prepare dataset for training."""
        logging.info("Loading and preparing dataset...")
        
        raw_dataset_dict = strategies['dataset'].read_and_shuffle_dataset(args.random_state)
        
        dataset_input = strategies['dataset'].prepare_dataset(
            raw_dataset_dict,
            args.split_type,
            args.batch_size,
            args.random_state,
            strategies['learning_task']
        )
        
        logging.info("Dataset preparation completed.")
        return dataset_input

    def _train_and_evaluate(self, strategies, dataset_input, comet_logger):
        """Execute model training and evaluation."""
        logging.info("Starting model training and evaluation...")
        
        strategies['training'].train_and_evaluate_model(
            self.strategy_creator,
            dataset_input,
            comet_logger
        )


if __name__ == '__main__':
    try:
        args = argument_parser()
        DeepResponse(args).main()
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user.")
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise SystemExit(1) from e
