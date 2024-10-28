""" Base Dataset Strategy """

import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from src.model.build.graph_neural_network.mpnn import convert_smiles_to_graph, prepare_batch


class BaseDatasetStrategy(ABC):
    """ Base dataset strategy """

    def __init__(self, data_path, evaluation_data_path=None):
        self.data_path = data_path
        self.evaluation_data_path = evaluation_data_path

    @abstractmethod
    def split_dataset(self, dataset, *args, **kwargs):
        """ Split dataset """
        pass

    @abstractmethod
    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    @abstractmethod
    def prepare_dataset(self, dataset, split_type, batch_size, random_state, learning_task_strategy):
        """ Prepare dataset """
        pass

    @abstractmethod
    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """
        pass

    @staticmethod
    def create_mpnn_and_conv_dataset(dataset_raw):
        """ Creating the dataset into two for Message Passing Neural Network (MPNN) and
            Convolutional Neural Network (CONV) """
        mpnn = dataset_raw[['drug_name', 'smiles']].drop_duplicates(subset='drug_name')
        conv = dataset_raw[['cell_line_name', 'cell_line_features']].drop_duplicates(subset='cell_line_name')

        return mpnn, conv

    @staticmethod
    def convert_conv_dataset(data):
        """ Convert conv dataset to optimized format """
        logging.info("Convert conv dataset is started.")
        return np.array([row for row in data])

    def tf_dataset_creator(self, x, y, batch_size, mpnn, conv, learning_task_strategy):
        """
        Create a batched and prefetched TensorFlow dataset, applying the learning task strategy as needed.

        Parameters:
        - x: Independent variables
        - y: Dependent variable
        - batch_size: Batch size
        - mpnn: MPNN dataset
        - conv: Conv dataset
        - learning_task_strategy: Strategy to process targets for specific learning tasks

        Returns:
        - atom_dim: Dimension of atom features
        - bond_dim: Dimension of bond features
        - x_conv_shape: Shape of the convolutional dataset
        - batched_dataset: TensorFlow batched and prefetched dataset
        """
        x_data = pd.DataFrame(x.astype('str'), columns=['drug_name', 'cell_line_name'])
        x_data = x_data.merge(mpnn).merge(conv)
        del mpnn, conv
        x_mpnn = convert_smiles_to_graph(x_data.smiles)
        x_conv = self.convert_conv_dataset(x_data.cell_line_features)
        del x_data
        
        y = learning_task_strategy.process_targets(y)
        batched_dataset = tf.data.Dataset.from_tensor_slices((x_conv, x_mpnn, y)). \
            batch(batch_size).map(prepare_batch, num_parallel_calls=-1).prefetch(tf.data.AUTOTUNE)

        return x_mpnn[0][0][0].shape[0], x_mpnn[1][0][0].shape[0], x_conv.shape, batched_dataset
