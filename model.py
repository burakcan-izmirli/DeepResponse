import pandas as pd  # 1.3.4
# import tensorflow as tf  # 2.6.0
# import numpy as np  # 1.19.5
import matplotlib.pyplot as plt
import warnings
from mpnn import *

warnings.filterwarnings('ignore')

# %%
dataset_raw = pd.read_pickle("burakcan_dataset.pkl")

dataset_raw = dataset_raw.sample(frac=1).reset_index(drop=True)

plt.hist(dataset_raw.pic50, bins = 50)
plt.show()

dataset_raw = dataset_raw.iloc[0:5000]

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
def convert_conv_dataset(dataset):
    last_list = []
    for i in dataset:
        dump_list = i.to_numpy()
        last_list.append(dump_list)

    return np.array(last_list)


x_train_conv = convert_conv_dataset(x_train_conv)
x_valid_conv = convert_conv_dataset(x_valid_conv)
x_test_conv = convert_conv_dataset(x_test_conv)


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
    conv_1 = layers.Conv2D(1, (4, 1), strides = 1, padding = 'same', activation = 'relu', name = 'conv_1')(input_layer)
    batch_norm_1 = layers.BatchNormalization()(conv_1)
    conv_2 = layers.Conv2D(1, (4, 1), strides = 1, padding = 'same', activation = 'relu', name = 'conv_2')(batch_norm_1)
    batch_norm_2 = layers.BatchNormalization()(conv_2)
    dropout_1 = layers.Dropout(0.1, name = 'dropout_1')(batch_norm_2)
    pooling_1 = layers.MaxPooling2D((2, 2), name = 'pooling_1')(dropout_1)
    conv_3 = layers.Conv2D(1, (1, 1), activation = 'relu', name = 'conv_3')(pooling_1)
    batch_norm_3 = layers.BatchNormalization()(conv_3)
    pooling_2 = layers.MaxPooling2D((2, 2), name = 'pooling_2')(batch_norm_3)
    flatten_1 = layers.Flatten()(pooling_2)

    concat = layers.concatenate([flatten_1, x], name = 'concat')

    dense_1 = layers.Dense(256, activation = 'relu')(concat)
    dense_2 = layers.Dense(128, activation = 'relu')(dense_1)
    dense_3 = layers.Dense(1, activation = 'linear')(dense_2)


    model = keras.Model(
        inputs = [input_layer, atom_features, bond_features, pair_indices, molecule_indicator],
        outputs = [dense_3],
        name = 'final_output'
    )

    return model


# %%
model = merged_model(
    atom_dim = x_train_mpnn[0][0][0].shape[0],
    bond_dim = x_train_mpnn[1][0][0].shape[0])

# %%
model.compile(
    loss = keras.losses.MeanSquaredError(),
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    # metrics=[keras.metrics.AUC(name="AUC")],
)

keras.utils.plot_model(model, show_dtype = True, show_shapes = True)

# %%
train_dataset = dataset_new(x_train_conv, x_train_mpnn, y_train, batch_size = 32)
valid_dataset = dataset_new(x_valid_conv, x_valid_mpnn, y_valid, batch_size = 32)

# %%
history = model.fit(
    train_dataset,
    validation_data = valid_dataset,
    epochs = 10,
    verbose = 1
)

