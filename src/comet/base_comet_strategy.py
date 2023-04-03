""" Base comet strategy """

from abc import ABC, abstractmethod


class BaseCometStrategy(ABC):
    """ Base comet strategy """

    @abstractmethod
    def integrate_comet(self):
        """ Integrate to comet to track the results"""
        pass
