"""
Main file of deep response
"""
import os
import numpy as np
import tensorflow as tf
from comet_ml import Experiment
from dotenv import load_dotenv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from src.data.prepare_dataset import prepare_dataset
from src.model.mpnn import create_mpnn_model
from src.model.conv import create_conv_model
from src.model.mlp import create_mlp_model
from src.util.static.default_arguments import DefaultArguments

tf.config.run_functions_eagerly(True)
# %%
# Arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--seed", default=DefaultArguments.seed.value, type=int, help="Seed")
parser.add_argument("-b", "--batch_size", default=DefaultArguments.batch_size.value, type=int, help="Batch Size")
parser.add_argument("-e", "--epoch", default=DefaultArguments.epoch.value, type=int, help="Epoch size")
parser.add_argument("-l", "--learning_rate", default=DefaultArguments.learning_rate.value, type=float,
                    help="Learning Rate")
parser.add_argument("-d", "--data_type", default=DefaultArguments.data_type.value, type=str, help="Data Type")
parser.add_argument("-c", "--comet", default=False, type=bool, help="Whether to use comet or not")
args = vars(parser.parse_args())

random_state = args["seed"]
batch_size = args["batch_size"]
learning_rate = args["learning_rate"]
epoch = args["epoch"]
data_type = args["data_type"]
use_comet = args["comet"]
# %%
# Creating Comet experiment to track results
if use_comet:
    load_dotenv('./dev.env')
    experiment = Experiment(
        api_key=os.environ.get("api_key"),
        project_name=os.environ.get("project_name"),
        workspace=os.environ.get("workspace"))
# %%
# Seeding everything to ensure reproducibility
np.random.seed(random_state)
tf.random.set_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

# %%
atom_dim, bond_dim, cell_line_dim, train_dataset, valid_dataset, test_dataset, y_test = prepare_dataset(data_type,
                                                                                                        batch_size,
                                                                                                        random_state)


# %%
def merged_model(cell_line_dims, atom_dims, bond_dims, batch_size=32, message_units=64, message_steps=4,
                 num_attention_heads=8,
                 dense_units=512):
    """
    Create merged model
    :param cell_line_dims: Cell line dimensions
    :param atom_dims: Atom dimensions
    :param bond_dims: Bond dimensions
    :param batch_size: Batch size
    :param message_units: Message units
    :param message_steps: Message steps
    :param num_attention_heads: Number of attention heads
    :param dense_units: Dense units
    :return: Merged model
    """
    atom_features, bond_features, pair_indices, molecule_indicator, mpnn = create_mpnn_model(atom_dims, bond_dims,
                                                                                             message_units,
                                                                                             message_steps,
                                                                                             num_attention_heads,
                                                                                             dense_units,
                                                                                             batch_size)
    cell_line_features, conv = create_conv_model(cell_line_dims)

    concat = keras.layers.concatenate([conv, mpnn], name='concat')

    mlp = create_mlp_model(dense_units, concat)

    model_func = keras.Model(inputs=[cell_line_features, atom_features, bond_features, pair_indices,
                                     molecule_indicator],
                             outputs=[mlp],
                             name='final_output')

    return model_func


# %%
model = merged_model(cell_line_dims=cell_line_dim, batch_size=batch_size, atom_dims=atom_dim, bond_dims=bond_dim)

# %%
model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# If you'd like to check the architecture of the model
# keras.utils.plot_model(model, show_dtype = True, show_shapes = True)

# Printing summary of the model
print(model.summary())
# %%
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epoch,
    verbose=1)

# %%
predictions = model.predict(test_dataset, verbose=1)
mse = mean_squared_error(y_test, predictions)
print(mse)
