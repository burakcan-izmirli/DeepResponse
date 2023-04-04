""" Skip comet strategy"""
import logging

from src.comet.base_comet_strategy import BaseCometStrategy


class SkipCometStrategy(BaseCometStrategy):
    """ Skip comet strategy """

    def integrate_comet(self):
        """ Using Comet to track results """
        logging.info("Comet integration was skipped.")
