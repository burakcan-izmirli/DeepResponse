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
#%%
dataset_raw = pd.read_csv("burakcan_dataset.csv")

plt.hist(dataset_raw.pic50, bins =50)
plt.show()

dataset_raw = dataset_raw.iloc[0:1000]


# print(dataset_raw.iloc[0:10,].smiles)


# csv_path = keras.utils.get_file(
#     "BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
# )
#
# dataset_raw2 = pd.read_csv(csv_path, usecols=[1, 2, 3])
# print(dataset_raw2.iloc[0:10,].smiles)
#%%
permuted_indices = np.random.permutation(np.arange(dataset_raw.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(dataset_raw.shape[0] * 0.8)]
x_train_mpnn = graphs_from_smiles(dataset_raw.iloc[train_index].smiles)
x_train_conv = dataset_raw.iloc[train_index].cell_line_features
y_train = dataset_raw.iloc[train_index].pic50

#%%
# Valid set: 19 % of data
valid_index = permuted_indices[int(dataset_raw.shape[0] * 0.8) : int(dataset_raw.shape[0] * 0.99)]
x_valid_mpnn = graphs_from_smiles(dataset_raw.iloc[valid_index].smiles)
x_valid_conv = dataset_raw.iloc[valid_index].cell_line_features
y_valid = dataset_raw.iloc[valid_index].pic50

#%%
# Test set: 1 % of data
test_index = permuted_indices[int(dataset_raw.shape[0] * 0.99) :]
x_test_mpnn = graphs_from_smiles(dataset_raw.iloc[test_index].smiles)
x_test_conv = dataset_raw.iloc[train_index].cell_line_features
y_test = dataset_raw.iloc[test_index].pic50


#%%
def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[x],
    )
    return model

# %%
def mergedModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    # x = layers.Dense(dense_units, activation="relu")(x)
    # x = layers.Dense(1, activation="sigmoid")(x)

    input_layer = layers.Input(shape=(3, 1), name='input_layer')
    conv_1 = layers.Conv1D(64, 2, strides=1, padding='same', activation='relu', name='conv_1')(input_layer)
    conv_2 = layers.Conv1D(64, 1, strides=1, padding='same', activation='relu', name='conv_2')(conv_1)
    dropout_1 = layers.Dropout(0.5, name= 'dropout_1')(conv_2)
    pooling_1 = layers.MaxPooling1D(1, name = 'pooling_1')(dropout_1)
    conv_3 = layers.Conv1D(64, 1, activation='relu', name='conv_3')(pooling_1)
    pooling_2 = layers.MaxPooling1D(2, name = 'pooling_2')(conv_3)
    flatten_1 = layers.Flatten()(pooling_2)

    concat = layers.concatenate([flatten_1, x], name='concat')

    dense_1 = layers.Dense(100, activation='relu')(concat)
    dense_2 = layers.Dense(16501, activation='sigmoid')(dense_1)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator, input_layer],
        outputs=[dense_2],
        name = 'final_output'
    )

    return model


# %%
mpnn = mergedModel(
    atom_dim=x_train_mpnn[0][0][0].shape[0], bond_dim=x_train_mpnn[1][0][0].shape[0])
# %%
mpnn.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate = 0.1)
    # metrics=[keras.metrics.AUC(name="AUC")],
)

keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)
#%%
train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)
#%%
history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=5,
    verbose=1
)
#%%
dump = pd.read_csv(io.StringIO(x_train_conv[0]), sep="\s\s+")[:-1]

#%%
