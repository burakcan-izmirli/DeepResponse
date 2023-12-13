""" Drug cell stratified dataset strategy """
import math
import numpy as np
import pandas as pd
import concurrent.futures

from helper.enum.dataset.n_split import NSplit

from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class DrugCellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Drug cell stratified dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """

        dataset_raw = pd.read_pickle(self.data_path)
        # Shuffling dataset
        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)

        evaluation_dataset_raw = None

        return {'dataset': dataset_raw, 'evaluation_dataset': evaluation_dataset_raw}

    def create_splitter(self, dataset, random_state):
        """ Create splitter """

        grouped_by_drug_df = dataset.drop_duplicates(subset=['drug_name'], keep='first')
        grouped_by_drug_df = grouped_by_drug_df[['drug_name']]
        grouped_by_drug_df['drug_group'] = np.divmod(np.arange(len(grouped_by_drug_df)),
                                                     math.ceil(len(grouped_by_drug_df)) /
                                                     NSplit.stratified.value)[0] + 1
        grouped_by_drug_df['drug_group'] = grouped_by_drug_df['drug_group'].astype('int')

        grouped_by_cell_df = dataset.drop_duplicates(subset=['cell_line_name'], keep='first')
        grouped_by_cell_df = grouped_by_cell_df[['cell_line_name']]
        grouped_by_cell_df['cell_group'] = np.divmod(np.arange(len(grouped_by_cell_df)),
                                                     math.ceil(len(grouped_by_cell_df)) /
                                                     NSplit.stratified.value)[0] + 1
        grouped_by_cell_df['cell_group'] = grouped_by_cell_df['cell_group'].astype('int')

        dataset = dataset.merge(grouped_by_drug_df, how='left').merge(grouped_by_cell_df, how='left'). \
            sort_values(['drug_group', 'cell_group']).reset_index(drop=True)

        for group in range(1, NSplit.stratified.value + 1):
            train_df = dataset.query('drug_group!=@group & cell_group!=@group')
            test_df = dataset.query('drug_group==@group | cell_group==@group')
            yield train_df, test_df

    def split_dataset(self, dataset, *args, **kwargs):
        """ Split dataset """
        return dataset[['drug_name', 'cell_line_name']], dataset[['pic50']]

    def prepare_dataset(self, dataset, split_type, batch_size, random_state):
        """
        Main function for preparing dataset
        :param dataset: Dataset
        :param split_type: Split type [random, cell_stratified, drug_stratified, cell_drug_stratified]
        :param batch_size: Batch size
        :param random_state: Random state
        :return: atom_dim, bond_dim, train_dataset, valid_dataset, test_dataset
        """
        dataset = dataset['dataset']

        mpnn_dataset, conv_dataset = self.create_mpnn_and_conv_dataset(dataset)

        dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]
        splitter = self.create_splitter(dataset, random_state)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Creating Tensorflow datasets in parallel for each train-test split
            futures = []

            for train_df, test_df in splitter:
                train_args = tuple(self.split_dataset(train_df)) + (batch_size, mpnn_dataset, conv_dataset)
                test_args = tuple(self.split_dataset(test_df)) + (batch_size, mpnn_dataset, conv_dataset)

                futures.append(executor.submit(self.tf_dataset_creator, *train_args))
                futures.append(executor.submit(self.tf_dataset_creator, *test_args))

            # Use as_completed to iterate over completed futures
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Unpack the results for the first train-test split
        atom_dim, bond_dim, cell_line_dim = results[0][:3]
        train_dataset, test_dataset = [result[3] for result in results]

        yield (atom_dim, bond_dim, cell_line_dim), train_dataset, test_dataset, results[1][-1]

