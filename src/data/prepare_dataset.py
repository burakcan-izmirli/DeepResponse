"""
Prepare dataset
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model.mpnn import graphs_from_smiles, prepare_batch

from src.util.enum.data_type import DataType
from src.util.enum.split_ratio import SplitRatio


def read_or_create_raw_dataset(data_type):
    """
    Reading or creating raw dataset
    """
    # Reading data according to the data type argument
    # Datasets were read from pickle file for faster execution.
    # Datasets can be created by calling related functions
    if data_type == DataType.normal.label:
        dataset_raw_df = pd.read_pickle(DataType.normal.path)
        # dataset_raw_df = create_dataset()
    elif data_type == DataType.l1000.label:
        dataset_raw_df = pd.read_pickle(DataType.l1000.path)
        # dataset_raw_df = create_l1000_dataset()
    elif data_type == DataType.pathway.label:
        dataset_raw_df = pd.read_pickle(DataType.pathway.path)
        # dataset_raw_df = create_pathway_sorted_dataset()
    elif data_type == DataType.pathway_reduced.label:
        dataset_raw_df = pd.read_pickle(DataType.pathway_reduced.path)
        # dataset_raw_df = create_pathway_sorted_reduced_dataset()
    elif data_type == DataType.digestive.label:
        dataset_raw_df = pd.read_pickle(DataType.digestive.path)
        # dataset_raw_df = create_tissue_dataset('digestive_system')
    else:
        print("Data type is wrong!")
        exit()

    return dataset_raw_df


def shuffle_split_dataset(dataset_raw, random_state):
    """
    Shuffle and split dataset into mpnn and conv
    :param dataset_raw:
    :param random_state:
    :return:
    """
    # Shuffling data
    dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)
    dataset = dataset_raw[['drug_name', 'cell_line_name', 'pic50']]

    # Splitting the data into two for Message Passing Neural Network (MPNN) and Convolutional Neural Network (CONV)
    mpnn = dataset_raw[['drug_name', 'smiles']].drop_duplicates()
    conv = dataset_raw[['cell_line_name', 'cell_line_features']].drop_duplicates(subset='cell_line_name')

    return dataset, mpnn, conv


def convert_conv_dataset(data):
    """
    Convert conv dataset to optimized format
    :param data: Raw conv dataset
    :return: Converted dataset
    """
    last_list = []
    for i in data:
        dump_list = i.to_numpy()
        last_list.append(dump_list)

    return np.array(last_list)


def tf_dataset_creator(x, y, batch_size, mpnn, conv):
    """
    Creating batched prefetched tensorflow dataset
    :param x: Independent variables
    :param y: Dependent variable
    :param batch_size: Batch size
    :param mpnn: MPNN data
    :param conv: Conv data
    :return: atom_dim, bond_dim and dataset
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices((x, (y)))

    batched_dataset = tf_dataset.batch(batch_size)
    for i in batched_dataset.as_numpy_iterator():
        x_data, y_data = i

    x_data = pd.DataFrame(x_data.astype('str'), columns=['drug_name', 'cell_line_name'])
    x_data = x_data.merge(mpnn).merge(conv)
    x_mpnn = graphs_from_smiles(x_data.smiles)
    x_conv = convert_conv_dataset(x_data.cell_line_features)
    batched_dataset = tf.data.Dataset.from_tensor_slices((x_conv, x_mpnn, (y_data))).batch(1)

    return x_mpnn[0][0][0].shape[0], x_mpnn[1][0][0].shape[0], x_conv.shape, \
        batched_dataset.map(prepare_batch, -1).prefetch(-1)


def prepare_dataset(data_type, batch_size, random_state):
    """
    Main function for preparing dataset
    :param data_type: Data type
    :param batch_size: Batch size
    :param random_state: Random state
    :return: atom_dim, bond_dim, train_dataset, valid_dataset, test_dataset
    """
    dataset_raw = read_or_create_raw_dataset(data_type)
    dataset, mpnn, conv = shuffle_split_dataset(dataset_raw, random_state)

    # Splitting data into train, validation and test
    x_train, x_test, y_train, y_test = train_test_split(dataset[['drug_name', 'cell_line_name']], dataset[['pic50']],
                                                        test_size=SplitRatio.test_ratio.value,
                                                        random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=SplitRatio.validation_ratio.value,
                                                      random_state=random_state)
    # Creating Tensorflow datasets
    atom_dim, bond_dim, cell_line_dim, train_dataset = tf_dataset_creator(x_train, y_train, batch_size, mpnn, conv)
    atom_dim_valid, bond_dim_valid, cell_line_dim_valid, valid_dataset = tf_dataset_creator(x_val, y_val, batch_size,
                                                                                            mpnn, conv)
    atom_dim_test, bond_dim_test, cell_line_dim_test, test_dataset = tf_dataset_creator(x_test, y_test, len(x_test),
                                                                                        mpnn, conv)

    return atom_dim, bond_dim, cell_line_dim, train_dataset, valid_dataset, test_dataset, y_test
