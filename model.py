import pandas as pd  # 1.3.4
import tensorflow as tf  # 2.6.0
import numpy as np  # 1.19.5
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv1D, Dense, Input, MaxPooling1D, Dropout, Flatten, concatenate
# from tensorflow.keras.utils import plot_model
# from IPython.display import Image
# from tensorflow.keras.optimizers import Adam
# from tensorflow import keras
# from tensorflow.keras import layers
# from rdkit import RDLogger
# from rdkit.Chem.Draw import IPythonConsole
# from rdkit.Chem.Draw import MolsToGridImage
import matplotlib.pyplot as plt
import warnings
import io

warnings.filterwarnings('ignore')

from mpnn import *

# %%
dataset_raw = pd.read_pickle("burakcan_dataset_v2.pkl")

plt.hist(dataset_raw.pic50, bins = 50)
plt.show()

dataset_raw = dataset_raw.iloc[0:40]

# print(dataset_raw.iloc[0:10,].smiles)


# csv_path = keras.utils.get_file(
#     "BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
# )
#
# dataset_raw2 = pd.read_csv(csv_path, usecols=[1, 2, 3])
# print(dataset_raw2.iloc[0:10,].smiles)
# %%
permuted_indices = np.random.permutation(np.arange(dataset_raw.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(dataset_raw.shape[0] * 0.8)]
x_train_mpnn = graphs_from_smiles(dataset_raw.iloc[train_index].smiles)
x_train_conv = dataset_raw.iloc[train_index].cell_line_features
y_train = dataset_raw.iloc[train_index].pic50

# %%
# Valid set: 19 % of data
valid_index = permuted_indices[int(dataset_raw.shape[0] * 0.8): int(dataset_raw.shape[0] * 0.99)]
x_valid_mpnn = graphs_from_smiles(dataset_raw.iloc[valid_index].smiles)
x_valid_conv = dataset_raw.iloc[valid_index].cell_line_features
y_valid = dataset_raw.iloc[valid_index].pic50

# %%
# Test set: 1 % of data
test_index = permuted_indices[int(dataset_raw.shape[0] * 0.99):]
x_test_mpnn = graphs_from_smiles(dataset_raw.iloc[test_index].smiles)
x_test_conv = dataset_raw.iloc[train_index].cell_line_features
y_test = dataset_raw.iloc[test_index].pic50


# %%
def mpnn_model(
        atom_dim,
        bond_dim,
        batch_size=32,
        message_units=64,
        message_steps=4,
        num_attention_heads=8,
        dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype = "float32", name = "atom_features")
    bond_features = layers.Input((bond_dim), dtype = "float32", name = "bond_features")
    pair_indices = layers.Input((2), dtype = "int32", name = "pair_indices")
    molecule_indicator = layers.Input((), dtype = "int32", name = "molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dense(dense_units, activation = "relu")(x)
    x = layers.Dense(1, activation = "sigmoid")(x)

    model = keras.Model(
        inputs = [atom_features, bond_features, pair_indices, molecule_indicator],
        outputs = [x],
    )
    return model

# %%
def conv_model():
    input_layer = layers.Input(shape = (16501, 4, 1), name = 'input_layer')
    conv_1 = layers.Conv2D(64, (1, 4), strides = 1, padding = 'same', activation = 'relu', name = 'conv_1')(input_layer)
    conv_2 = layers.Conv2D(64, (1, 4), strides = 1, padding = 'same', activation = 'relu', name = 'conv_2')(conv_1)
    dropout_1 = layers.Dropout(0.5, name = 'dropout_1')(conv_2)
    pooling_1 = layers.MaxPooling2D((1, 1), name = 'pooling_1')(dropout_1)
    conv_3 = layers.Conv2D(64, (1, 1), activation = 'relu', name = 'conv_3')(pooling_1)
    pooling_2 = layers.MaxPooling2D((2, 2), name = 'pooling_2')(conv_3)
    flatten_1 = layers.Flatten()(pooling_2)

    dense_1 = layers.Dense(1, activation = 'relu')(flatten_1)
    # dense_2 = layers.Dense(16501, activation = 'sigmoid')(dense_1)

    model = keras.Model(
        inputs = [input_layer],
        outputs = [dense_1],
        name = 'final_output'
    )

    return model
# %%
def merged_model(
        atom_dim,
        bond_dim,
        batch_size=32,
        message_units=64,
        message_steps=4,
        num_attention_heads=8,
        dense_units=512,
):
    atom_features = layers.Input((atom_dim), dtype = "float32", name = "atom_features")
    bond_features = layers.Input((bond_dim), dtype = "float32", name = "bond_features")
    pair_indices = layers.Input((2), dtype = "int32", name = "pair_indices")
    molecule_indicator = layers.Input((), dtype = "int32", name = "molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    # x = layers.Dense(dense_units, activation="relu")(x)
    # x = layers.Dense(1, activation="sigmoid")(x)

    input_layer = layers.Input(shape = (16501, 4, 1), name = 'input_layer')
    conv_1 = layers.Conv2D(64, (1, 4), strides = 1, padding = 'same', activation = 'relu', name = 'conv_1')(input_layer)
    conv_2 = layers.Conv2D(64, (1, 4), strides = 1, padding = 'same', activation = 'relu', name = 'conv_2')(conv_1)
    dropout_1 = layers.Dropout(0.5, name = 'dropout_1')(conv_2)
    pooling_1 = layers.MaxPooling2D((1, 1), name = 'pooling_1')(dropout_1)
    conv_3 = layers.Conv2D(64, (1, 1), activation = 'relu', name = 'conv_3')(pooling_1)
    pooling_2 = layers.MaxPooling2D((2, 2), name = 'pooling_2')(conv_3)
    flatten_1 = layers.Flatten()(pooling_2)

    concat = layers.concatenate([flatten_1, x], name = 'concat')

    dense_1 = layers.Dense(1, activation = 'relu')(concat)

    model = keras.Model(
        inputs = [input_layer, atom_features, bond_features, pair_indices, molecule_indicator],
        outputs = [dense_1],
        name = 'final_output'
    )

    return model


# %%
mpnn = merged_model(
    atom_dim = x_train_mpnn[0][0][0].shape[0], bond_dim = x_train_mpnn[1][0][0].shape[0])
mpnn_2 = mpnn_model(
    atom_dim = x_train_mpnn[0][0][0].shape[0], bond_dim = x_train_mpnn[1][0][0].shape[0])
mpnn_3 = conv_model()

mpnn = mpnn
# %%
mpnn.compile(
    loss = keras.losses.MeanSquaredError(),
    optimizer = keras.optimizers.Adam(learning_rate = 0.1)
    # metrics=[keras.metrics.AUC(name="AUC")],
)

keras.utils.plot_model(mpnn, show_dtype = True, show_shapes = True)
# %%
# train_dataset_mpnn = mpnn_dataset(x_train_mpnn,  y_train, batch_size = 1)
# valid_dataset_mpnn = mpnn_dataset(x_valid_mpnn, y_valid)
# test_dataset_mpnn = mpnn_dataset(x_test_mpnn, y_test)
# %%
# last_list = []
# for i in x_train_conv:
#     for j in range(0, len(x_train_conv)):
#         dump_list = []
#         dump_list.append(i.iloc[j])
#     last_list.append(dump_list)
#
# b = tf.ragged.constant(last_list, dtype = tf.float32)
# %%
last_list = []
for i in x_train_conv:
    dump_list = i.to_numpy()
    last_list.append(dump_list)

x_train_conv_new = np.array(last_list)
# %%
train_dataset_mpnn = dataset_new(x_train_conv_new, x_train_mpnn,  y_train, batch_size = 1)
# %%
# train_dataset_conv = conv_dataset(b, y_train, batch_size = 1)
# valid_dataset_conv = conv_dataset(x_train_conv, y_valid)
# test_dataset_conv = conv_dataset(x_train_conv, y_test)

# %%
history = mpnn.fit(
    train_dataset_mpnn,
    epochs = 5,
    verbose = 1
)

