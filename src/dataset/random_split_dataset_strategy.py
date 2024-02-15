""" Random split dataset strategy """
import pandas as pd
import numpy as np
import concurrent.futures

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helper.enum.dataset.split_ratio import SplitRatio
from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class RandomSplitDatasetStrategy(BaseDatasetStrategy):
    """ Random split dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """

        dataset_raw = pd.read_pickle(self.data_path)

        # Shuffling dataset
        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # evaluation_dataset_raw = pd.read_pickle(self.evaluation_data_path)
        # evaluation_dataset_raw = evaluation_dataset_raw.sample(frac=1, random_state=random_state). \
        #     reset_index(drop=True)

        return {'dataset': dataset_raw}

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    # def split_dataset(self, dataset, random_state):
    #     """ Splitting dataset as train, validation and test """
    #
    #     x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(dataset[['drug_name', 'cell_line_name']],
    #                                                                     dataset[['pic50']],
    #                                                                     test_size=SplitRatio.test_ratio.value,
    #                                                                     random_state=random_state)
    #
    #     x_train_df, x_val_df, y_train_df, y_val_df = train_test_split(x_train_df,
    #                                                                   y_train_df,
    #                                                                   test_size=SplitRatio.validation_ratio.value,
    #                                                                   random_state=random_state)
    #     return x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df

    def split_dataset(self, dataset, random_state):
        """ Splitting dataset as train, validation and test """

        # First split: separate out the test set
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            dataset[['drug_name', 'cell_line_name', 'cancer_type']],
            dataset[['pic50']],
            test_size=SplitRatio.test_ratio.value,
            stratify=dataset['cancer_type'],
            random_state=random_state)

        # Second split: separate out the training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                          test_size=SplitRatio.validation_ratio.value / (
                                                                  1 - SplitRatio.test_ratio.value),
                                                          stratify=x_train_val['cancer_type'],
                                                          random_state=random_state)

        x_train = x_train.drop(columns=['cancer_type'])
        x_val = x_val.drop(columns=['cancer_type'])
        x_test = x_test.drop(columns=['cancer_type'])

        return x_train, x_val, x_test, y_train, y_val, y_test

    def prepare_dataset(self, dataset, split_type, batch_size, random_state):
        """
        Main function for preparing dataset.

        :param dataset: Dataset
        :param split_type: Split type [random, cell_stratified, drug_stratified, cell_drug_stratified]
        :param batch_size: Batch size
        :param random_state: Random state
        :return: Tuple containing atom_dim, bond_dim, cell_line_dim, train_datasets, valid_datasets, test_datasets, y_test
        """
        dataset = dataset['dataset']
        mpnn_dataset, conv_dataset = self.create_mpnn_and_conv_dataset(dataset)
        # dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]
        dataset = dataset[['drug_name', 'cell_line_name', 'pic50', 'cancer_type']]

        # Splitting dataset into train, validation, and test
        x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(dataset, random_state)

        # scaler = StandardScaler()
        #
        # # Concatenate all the arrays in 'cell_line_features' in the training data
        # train_arrays = np.concatenate(
        #     conv_dataset.query('cell_line_name in @x_train.cell_line_name.tolist()')['cell_line_features'].values)
        # # Fit the scaler on the concatenated data
        # scaler.fit(train_arrays)
        #
        # # Define a function to transform each array in 'cell_line_features'
        # def transform_features(features):
        #     return scaler.transform(features)
        #
        # # Apply the function to the 'cell_line_features' in each DataFrame
        # x_train_indices = conv_dataset.query('cell_line_name in @x_train.cell_line_name.tolist()').index
        # x_val_indices = conv_dataset.query('cell_line_name in @x_val.cell_line_name.tolist()').index
        # x_test_indices = conv_dataset.query('cell_line_name in @x_test.cell_line_name.tolist()').index
        #
        # conv_dataset.loc[x_train_indices, 'cell_line_features'] = conv_dataset.loc[
        #     x_train_indices, 'cell_line_features'].apply(transform_features)
        # conv_dataset.loc[x_val_indices, 'cell_line_features'] = conv_dataset.loc[
        #     x_val_indices, 'cell_line_features'].apply(transform_features)
        # conv_dataset.loc[x_test_indices, 'cell_line_features'] = conv_dataset.loc[
        #     x_test_indices, 'cell_line_features'].apply(transform_features)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Creating Tensorflow datasets in parallel
            futures = [
                executor.submit(self.tf_dataset_creator, x_train, y_train, batch_size, mpnn_dataset, conv_dataset),
                executor.submit(self.tf_dataset_creator, x_val, y_val, batch_size, mpnn_dataset, conv_dataset),
                executor.submit(self.tf_dataset_creator, x_test, y_test, batch_size, mpnn_dataset, conv_dataset)
            ]

            results = [future.result() for future in futures]

        # Unpack the results
        atom_dim, bond_dim, cell_line_dim = results[0][:3]
        train_dataset, valid_dataset, test_dataset = [result[3] for result in results]

        return (atom_dim, bond_dim, cell_line_dim), train_dataset, valid_dataset, test_dataset, y_test
