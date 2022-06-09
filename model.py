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

from mpnn import *
#%%
dataset_raw = pd.read_csv("burakcan_dataset.csv")
dataset_raw = dataset_raw.iloc[0:50]
#%%
permuted_indices = np.random.permutation(np.arange(dataset_raw.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(dataset_raw.shape[0] * 0.8)]
x_train = graphs_from_smiles(dataset_raw.iloc[train_index].smiles)
y_train = dataset_raw.iloc[train_index].pic50

#%%
# Valid set: 19 % of data
valid_index = permuted_indices[int(dataset_raw.shape[0] * 0.8) : int(dataset_raw.shape[0] * 0.99)]
x_valid = graphs_from_smiles(dataset_raw.iloc[valid_index].smiles)
y_valid = dataset_raw.iloc[valid_index].pic50

#%%
# Test set: 1 % of data
test_index = permuted_indices[int(dataset_raw.shape[0] * 0.99) :]
x_test = graphs_from_smiles(dataset_raw.iloc[test_index].smiles)
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
mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)
# %%
mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.AUC(name="AUC")],
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
    verbose=1,
    class_weight={0: 2.0, 1: 0.5},
)

#%%
