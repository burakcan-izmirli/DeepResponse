"""
DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response 
with Deep Learning Based Pharmacogenomic Modelling

CLI entry point for training and evaluating DeepResponse models.

This script loads a processed pharmacogenomic dataset (e.g., DepMap/CCLE/GDSC),
builds TensorFlow datasets from drug molecular features (SMILES) and multi-omics
cell-line features, and runs the configured split strategy (random/stratified/
cross-domain) for regression or classification.
"""
import os
import sys
import time
import logging
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from helper.argument_parser import argument_parser

_LOG_FORMAT = '%(levelname)s:%(name)s:%(asctime)s:%(message)s'

class DeepResponse:
    def __init__(self, args: Namespace):
        from src.strategy_creator import StrategyCreator
        self.strategy_creator = StrategyCreator(args)

    def _configure_logging(self, args: Namespace):
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_parts = [args.data_source]
        if args.evaluation_source:
            log_parts.append(f"to_{args.evaluation_source}")
        log_parts.extend([args.split_type, args.learning_task, f"stl{args.selformer_trainable_layers}"])
        log_path = log_dir / f"{'_'.join(map(str, log_parts))}.log"

        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ]
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=handlers, force=True)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.info("Logging to %s", log_path)

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
            args = self.strategy_creator.args
            self._configure_logging(args)
            logging.info("DeepResponse started.")

            # Initialize Comet logging
            comet_logger = self._initialize_comet_logger()

            mixed_precision_enabled = False
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices("GPU"):
                    tf.keras.mixed_precision.set_global_policy("mixed_float16")
                    mixed_precision_enabled = True
                    logging.info("Mixed precision enabled (mixed_float16).")
            except ModuleNotFoundError:
                pass
            except Exception as exc:
                logging.warning(f"Could not enable mixed precision: {exc}")

            self._log_configuration(args, mixed_precision_enabled)

            # Set random seed for reproducibility
            from helper.seed_setter import set_seed
            set_seed(args.random_state)

            # Initialize strategies
            strategies = self._initialize_strategies()
            
            # Load and prepare dataset
            dataset_input = self._prepare_dataset(strategies, args)
            
            # Train and evaluate model
            self._train_and_evaluate(strategies, dataset_input, comet_logger)

            logging.info("DeepResponse completed successfully.")

    def _initialize_comet_logger(self):
        """Initialize Comet ML logging strategy."""
        strategy = self.strategy_creator.get_comet_strategy()
        logger = strategy.integrate_comet()
        if logger is None and self.strategy_creator.use_comet:
            logging.warning("Comet requested but unavailable; proceeding without experiment logging.")
        return logger

    def _log_configuration(self, args, mixed_precision_enabled: bool):
        """Log current configuration parameters."""
        effective_lr = self.strategy_creator.get_effective_learning_rate()
        requested_norm = getattr(args, "cell_feature_normalization", "auto")
        effective_norm = requested_norm
        if requested_norm == "auto":
            effective_norm = "zscore" if args.split_type == "cross_domain" else "none"

        sections = {
            "Data": {
                "data_source": args.data_source,
                "evaluation_source": args.evaluation_source or "None",
                "data_type": args.data_type,
                "split_type": args.split_type,
                "learning_task": args.learning_task,
            },
            "Training": {
                "learning_rate": args.learning_rate,
                "effective_learning_rate": effective_lr,
                "epochs": args.epoch,
                "batch_size": args.batch_size,
                "random_state": args.random_state,
                "mixed_precision": mixed_precision_enabled,
            },
            "Model": {
                "selformer_trainable_layers": args.selformer_trainable_layers,
                "unfreeze_epoch": args.unfreeze_epoch,
                "unfreeze_layers": args.unfreeze_layers,
                "unfreeze_lr_factor": args.unfreeze_lr_factor,
            },
            "Pipeline": {
                "cache_datasets": args.cache_datasets,
                "cell_feature_normalization": (
                    effective_norm if requested_norm == effective_norm else f"{requested_norm} -> {effective_norm}"
                ),
            },
            "Logging": {
                "use_comet": args.use_comet,
            },
        }

        lines = ["==================== DeepResponse Configuration ===================="]
        for section_name, entries in sections.items():
            lines.append(f"[{section_name}]")
            key_width = max(len(key) for key in entries.keys())
            for key, value in entries.items():
                lines.append(f"  {key:<{key_width}} : {value}")
        lines.append("====================================================================")
        logging.info("\n".join(lines))

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
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        logging.error(f"Dataset file not found: {e}")
        raise SystemExit(1) from e
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise
    except Exception as e:
        logging.error(f"Unexpected error during execution: {e}")
        raise SystemExit(1) from e
