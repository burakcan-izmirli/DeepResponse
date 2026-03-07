"""
DeepResponse: Large Scale Prediction of Cancer Cell Line Drug Response
with Deep Learning Based Pharmacogenomic Modelling

CLI entry point for DeepResponse.
"""

import os

# Suppress TensorFlow/Transformers noise at startup.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging

from src.pipeline import DeepResponsePipeline
from utils.argument_parser import argument_parser
from utils.logger import setup_logging


if __name__ == "__main__":
    try:
        args = argument_parser()
        setup_logging(args)
        DeepResponsePipeline(args).execute()
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
        logging.exception("Unexpected error during execution:")
        raise SystemExit(1) from e
