""" Use comet strategy"""
from comet_ml import Experiment
import logging
import os
from dotenv import load_dotenv

from src.comet.base_comet_strategy import BaseCometStrategy


class UseCometStrategy(BaseCometStrategy):
    """ Use comet strategy """

    def integrate_comet(self):
        """ Using Comet to monitor results """
        logging.info("Comet was integrated successfully.")
        load_dotenv('./dev.env')
        return Experiment(api_key=os.environ.get("api_key"),
                          project_name=os.environ.get("project_name"),
                          workspace=os.environ.get("workspace"),
                          auto_histogram_tensorboard_logging=True,
                          auto_histogram_weight_logging=True,
                          auto_histogram_gradient_logging=True,
                          auto_histogram_activation_logging=True)
