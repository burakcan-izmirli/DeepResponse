""" Base dataset strategy """
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from src.model.model_creation.mpnn import graphs_from_smiles, prepare_batch


class BaseDatasetStrategy(ABC):
    """ Base dataset src """

    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def split_dataset(self, dataset, *args, **kwargs):
        """ Split dataset """
        pass

    @abstractmethod
    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    @abstractmethod
    def prepare_dataset(self, dataset, split_type, batch_size, random_state):
        """ Prepare dataset """
        pass

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """
        # Shuffling dataset
        dataset_raw = pd.read_pickle(self.data_path)
        return dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)

    @staticmethod
    def create_mpnn_and_conv_dataset(dataset_raw):
        """ Creating the dataset into two for Message Passing Neural Network (MPNN) and
            Convolutional Neural Network (CONV) """
        mpnn = dataset_raw[['drug_name', 'smiles']].drop_duplicates(subset='drug_name')
        conv = dataset_raw[['cell_line_name', 'cell_line_features']].drop_duplicates(subset='cell_line_name')

        return mpnn, conv

    @staticmethod
    def convert_conv_dataset(data):
        """
        Convert conv dataset to optimized format
        :param data: Raw conv dataset
        :return: Converted dataset
        """
        last_list = []
        for row in data:
            dump_list = row.to_numpy()
            last_list.append(dump_list)

        return np.array(last_list)

    def tf_dataset_creator(self, x, y, batch_size, mpnn, conv):
        """
        Creating batched prefetched tensorflow dataset
        :param x: Independent variables
        :param y: Dependent variable
        :param batch_size: Batch size
        :param mpnn: MPNN dataset
        :param conv: Conv dataset
        :return: atom_dim, bond_dim and dataset
        """
        tf_dataset = tf.data.Dataset.from_tensor_slices((x, (y)))

        batched_dataset = tf_dataset.batch(batch_size)
        for i in batched_dataset.as_numpy_iterator():
            x_data, y_data = i

        x_data = pd.DataFrame(x_data.astype('str'), columns=['drug_name', 'cell_line_name'])
        x_data = x_data.merge(mpnn).merge(conv)
        x_mpnn = graphs_from_smiles(x_data.smiles)
        x_conv = self.convert_conv_dataset(x_data.cell_line_features)
        batched_dataset = tf.data.Dataset.from_tensor_slices((x_conv, x_mpnn, (y_data))).batch(1)
        return x_mpnn[0][0][0].shape[0], x_mpnn[1][0][0].shape[0], x_conv.shape, \
            batched_dataset.map(prepare_batch, -1).prefetch(-1)
