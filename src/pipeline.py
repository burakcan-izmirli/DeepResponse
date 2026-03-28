"""Core DeepResponse pipeline orchestration."""

import logging
from argparse import Namespace
from typing import Any, Dict

from utils.logger import execution_timer, log_configuration


class DeepResponsePipeline:
    """End-to-end training/evaluation pipeline for DeepResponse."""

    def __init__(self, args: Namespace) -> None:
        from src.strategy_resolver import StrategyResolver

        self.args = args
        self.strategy_creator = StrategyResolver(args)

    def execute(self) -> None:
        """
        Execute the main DeepResponse pipeline.

        Raises:
            SystemExit: On any critical error during execution.
        """
        with execution_timer():
            logging.info("DeepResponse started.")

            comet_logger = self._initialize_comet_logger()

            log_configuration(self.args)

            from utils.seed_setter import (
                set_seed,
            )

            set_seed(self.args.random_state)

            strategies = self._initialize_strategies()

            dataset_input = self._prepare_dataset(strategies)

            self._train_and_evaluate(strategies, dataset_input, comet_logger)

            logging.info("DeepResponse completed successfully.")

    def _initialize_comet_logger(self) -> Any:
        """Initialize Comet ML logging strategy."""
        strategy = self.strategy_creator.get_comet_strategy()
        logger = strategy.integrate_comet()
        if logger is None and self.strategy_creator.use_comet:
            logging.warning(
                "Comet requested but unavailable; proceeding without experiment logging."
            )
        return logger

    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all required strategies."""
        split_strategy_dict = self.strategy_creator.get_split_strategy()

        return {
            "dataset": split_strategy_dict["dataset"],
            "training": split_strategy_dict["training"],
        }

    def _prepare_dataset(self, strategies: Dict[str, Any]) -> Any:
        """Load and prepare dataset for training."""
        logging.info("Loading and preparing dataset...")

        raw_dataset_dict = strategies["dataset"].read_and_shuffle_dataset(
            self.args.random_state
        )

        dataset_input = strategies["dataset"].prepare_dataset(
            raw_dataset_dict,
            self.args.split_type,
            self.args.batch_size,
            self.args.random_state,
        )

        logging.info("Dataset preparation completed.")
        return dataset_input

    def _train_and_evaluate(
        self,
        strategies: Dict[str, Any],
        dataset_input: Any,
        comet_logger: Any,
    ) -> None:
        """Execute model training and evaluation."""
        logging.info("Starting model training and evaluation...")

        strategies["training"].train_and_evaluate_model(
            self.strategy_creator, dataset_input, comet_logger
        )
