""" Random Split Dataset Strategy """

import pandas as pd
import concurrent.futures
from sklearn.model_selection import train_test_split

from helper.enum.dataset.binary_threshold import BinaryThreshold 
from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class RandomSplitDatasetStrategy(BaseDatasetStrategy):
    """ Random split dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """

        dataset_raw = pd.read_pickle(self.data_path)
        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # evaluation_dataset_raw = pd.read_pickle(self.evaluation_data_path)
        # evaluation_dataset_raw = evaluation_dataset_raw.sample(frac=1, random_state=random_state). \
        #     reset_index(drop=True)

        return {'dataset': dataset_raw}

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    def split_dataset(self, dataset, random_state):
        """ Splitting dataset as train, validation, and test """

        x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
            dataset[['drug_name', 'cell_line_name']], dataset[['pic50']],
            test_size=SplitRatio.test_ratio.value, random_state=random_state
        )

        x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(
            x_train_df, y_train_df,
            test_size=SplitRatio.validation_ratio.value, random_state=random_state
        )
        return x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df

    def prepare_dataset(self, dataset, split_type, batch_size, random_state, learning_task_strategy):
        """
        Main function for preparing dataset with learning task strategy.

        :param dataset: Dataset
        :param split_type: Split type [random, cell_stratified, drug_stratified, cell_drug_stratified]
        :param batch_size: Batch size
        :param random_state: Random state
        :param learning_task_strategy: Strategy for specific learning task
        :return: Tuple containing atom_dim, bond_dim, cell_line_dim, train_datasets, valid_datasets, test_datasets, y_test
        """
        dataset = dataset['dataset']
        mpnn_dataset, conv_dataset = self.create_mpnn_and_conv_dataset(dataset)
        dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]

        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(dataset, random_state)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.tf_dataset_creator, x_train, y_train, batch_size, mpnn_dataset, conv_dataset, learning_task_strategy),
                executor.submit(self.tf_dataset_creator, x_val, y_val, batch_size, mpnn_dataset, conv_dataset, learning_task_strategy),
                executor.submit(self.tf_dataset_creator, x_test, y_test, batch_size, mpnn_dataset, conv_dataset, learning_task_strategy)
            ]

            results = [future.result() for future in futures]

        atom_dim, bond_dim, cell_line_dim = results[0][:3]
        train_dataset, valid_dataset, test_dataset = [result[3] for result in results]

        return (atom_dim, bond_dim, cell_line_dim), train_dataset, valid_dataset, test_dataset, y_test
