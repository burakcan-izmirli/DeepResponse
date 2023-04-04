""" Seed setter """
import os
import numpy as np
import tensorflow as tf


def set_seed(random_state):
    """ Seeding everything to ensure reproducibility """
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

