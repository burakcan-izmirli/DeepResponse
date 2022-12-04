from mpnn import *
# %%
load_dotenv('dev.env')
experiment = Experiment(
    api_key=os.environ.get("api_key"),
    project_name="general",
    workspace="burakcan-izmirli",
    log_code = True
)
# %%
parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--seed", default = 12, type = int, help = "Seed")
parser.add_argument("-b", "--batch_size", default = 64, type = int, help = "Seed")
parser.add_argument("-e", "--epoch", default = "50", type = int, help = "Epoch size")
parser.add_argument("-l", "--learning_rate", default = 0.01, type = float, help = "Learning rate")
args = vars(parser.parse_args())

random_state = args["seed"]
batch_size = args["batch_size"]
learning_rate = args["learning_rate"]
epoch = args["epoch"]

# %%
# random_state = 12
# batch_size = 64
# learning_rate = 0.01
# epoch = 10
# %%
np.random.seed(random_state)
tf.random.set_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

dataset_raw = pd.read_pickle("burakcan_dataset_l1000.pkl")
dataset_raw = dataset_raw.sample(frac = 1, random_state = random_state).reset_index(drop = True)
dataset = dataset_raw[['drug_name', 'cell_line_name', 'pic50']]
mpnn = dataset_raw[['drug_name', 'smiles']].drop_duplicates()
conv = dataset_raw[['cell_line_name', 'cell_line_features']].drop_duplicates(subset = 'cell_line_name')
# %%
x_train, x_test, y_train, y_test = train_test_split(dataset[['drug_name', 'cell_line_name']], dataset[['pic50']],
                                                    test_size = 0.01, random_state = random_state)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.19, random_state = random_state)

# %%
atom_dim, bond_dim, train_dataset = dataset_creator(x_train, y_train, batch_size, mpnn, conv)
atom_dim_valid, bond_dim_valid, valid_dataset = dataset_creator(x_val, y_val, batch_size, mpnn, conv)
atom_dim_test, bond_dim_test, test_dataset = dataset_creator(x_test, y_test, len(x_test), mpnn, conv)


# %%
def merged_model(
        atom_dims,
        bond_dims,
        batch_size=32,
        message_units=64,
        message_steps=4,
        num_attention_heads=8,
        dense_units=512,
):
    atom_features = layers.Input((atom_dims), dtype = "float32", name = "atom_features")
    bond_features = layers.Input((bond_dims), dtype = "float32", name = "bond_features")
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

    input_layer = layers.Input(shape = (897, 4, 1), name = 'input_layer')
    conv_1 = layers.Conv2D(1, (4, 1), name = 'conv_1')(input_layer)
    # batch_norm_1 = layers.BatchNormalization()(conv_1)
    conv_2 = layers.Conv2D(1, (4, 1), name = 'conv_2')(conv_1)
    batch_norm_2 = layers.BatchNormalization()(conv_2)
    # dropout_1 = layers.Dropout(0.1, name = 'dropout_1')(batch_norm_2)
    # pooling_1 = layers.AveragePooling2D((4, 1), name = 'pooling_1')(batch_norm_2)
    # conv_3 = layers.Conv2D(1, (1, 1), activation = 'relu', name = 'conv_3')(pooling_1)
    # batch_norm_3 = layers.BatchNormalization()(pooling_1)
    # pooling_2 = layers.MaxPooling2D((2, 2), name = 'pooling_2')(batch_norm_3)
    flatten_1 = layers.GlobalAveragePooling2D()(batch_norm_2)

    concat = layers.concatenate([flatten_1, x], name = 'concat')

    dense_1 = layers.Dense(dense_units, activation = 'relu')(concat)
    dropout_1 = layers.Dropout(0.2, name = 'dropout_1')(dense_1)
    dense_2 = layers.Dense(128, activation = 'relu')(dropout_1)
    dense_3 = layers.Dense(1, activation = 'linear')(dense_2)

    model_func = keras.Model(inputs = [input_layer, atom_features, bond_features, pair_indices, molecule_indicator],
                             outputs = [dense_3],
                             name = 'final_output')

    return model_func


# %%
model = merged_model(
    batch_size = batch_size,
    atom_dims = atom_dim,
    bond_dims = bond_dim)

# %%
model.compile(
    loss = keras.losses.MeanSquaredError(),
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate)
    # metrics=[keras.metrics.AUC(name="AUC")],
)

# keras.utils.plot_model(model, show_dtype = True, show_shapes = True)

print(model.summary())
# %%
csv_logger = keras.callbacks.CSVLogger('log.csv', append = True, separator = ';')
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(
    train_dataset,
    validation_data = valid_dataset,
    epochs = epoch,
    verbose = 1,
    callbacks = [csv_logger]
)
# %%
predictions = model.predict(test_dataset, verbose = 1)
mse = mean_squared_error(y_test, predictions)
print(mse)