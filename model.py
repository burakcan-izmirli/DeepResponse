import pandas as pd
import numpy as np
from os import walk
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense, Input, MaxPooling1D, Dropout, Flatten, concatenate
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from tensorflow.keras.optimizers import Adam

from mpnn import *
#%%
filenames = next(walk("dataset/full"), (None, None, []))[2]
print(filenames)
#%%
