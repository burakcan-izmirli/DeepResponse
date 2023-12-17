""" Cell stratified dataset strategy """
import math
import pandas as pd
import numpy as np
import concurrent.futures

from sklearn.model_selection import LeaveOneGroupOut

from helper.enum.dataset.n_split import NSplit

from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class CellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Cell stratified dataset strategy """

    def read_and_shuffle_dataset(self, random_state):
        """ Read and shuffle dataset """

        dataset_raw = pd.read_pickle(self.data_path)
        # Shuffling dataset
        dataset_raw = dataset_raw.sample(frac=1, random_state=random_state).reset_index(drop=True)

        evaluation_dataset_raw = None

        return {'dataset': dataset_raw, 'evaluation_dataset': evaluation_dataset_raw}

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        grouped_by_cell_df = dataset.drop_duplicates(subset=['cell_line_name'], keep='first')
        grouped_by_cell_df = grouped_by_cell_df[['cell_line_name']]
        grouped_by_cell_df['group'] = np.divmod(np.arange(len(grouped_by_cell_df)),
                                                math.ceil(len(grouped_by_cell_df)) / NSplit.stratified.value)[0] + 1
        grouped_by_cell_df['group'] = grouped_by_cell_df['group'].astype('int')

        dataset = pd.merge(dataset, grouped_by_cell_df, how='outer').sort_values('group').reset_index(drop=True)
        leave_one_group_out = LeaveOneGroupOut()

        return dataset, leave_one_group_out.split(dataset[['drug_name', 'cell_line_name']], dataset[['pic50']],
                                                  groups=dataset['group'])

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

        dataset, splitter = self.create_splitter(dataset, random_state)

        for train, test in splitter:
            train_df = dataset[dataset.index.isin(train)]
            x_train, y_train = self.split_dataset(train_df)

            test_df = dataset[dataset.index.isin(test)]
            x_test, y_test = self.split_dataset(test_df)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Creating Tensorflow datasets in parallel
                futures = [
                    executor.submit(self.tf_dataset_creator, x_train, y_train, batch_size, mpnn_dataset, conv_dataset),
                    executor.submit(self.tf_dataset_creator, x_test, y_test, batch_size, mpnn_dataset, conv_dataset)
                ]

                results = [future.result() for future in futures]

            # Unpack the results
            atom_dim, bond_dim, cell_line_dim = results[0][:3]
            train_dataset, test_dataset = [result[3] for result in results]

            yield (atom_dim, bond_dim, cell_line_dim), train_dataset, test_dataset, y_test

            del train_df, x_train, y_train, test_df, x_test, y_test
