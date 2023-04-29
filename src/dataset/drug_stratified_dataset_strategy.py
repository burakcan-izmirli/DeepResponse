""" Drug stratified dataset strategy """
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from helper.enum.dataset.n_split import NSplit

from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class DrugStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Drug stratified dataset strategy """

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        grouped_by_drug_df = dataset.drop_duplicates(subset=['drug_name'], keep='first')
        grouped_by_drug_df = grouped_by_drug_df[['drug_name']]
        grouped_by_drug_df['group'] = np.divmod(np.arange(len(grouped_by_drug_df)),
                                                math.ceil(len(grouped_by_drug_df)) / NSplit.stratified.value)[0] + 1
        grouped_by_drug_df['group'] = grouped_by_drug_df['group'].astype('int')

        dataset = pd.merge(dataset, grouped_by_drug_df, how='outer').sort_values('group').reset_index(drop=True)
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
        mpnn_dataset, conv_dataset = self.create_mpnn_and_conv_dataset(dataset)

        dataset = dataset[['drug_name', 'cell_line_name', 'pic50']]
        dataset, splitter = self.create_splitter(dataset, random_state)
        for train, test in splitter:
            train_df = dataset[dataset.index.isin(train)]
            x_train, y_train = self.split_dataset(train_df)
            # Creating Tensorflow datasets
            atom_dim, bond_dim, cell_line_dim, train_dataset = self.tf_dataset_creator(x_train, y_train, batch_size,
                                                                                       mpnn_dataset, conv_dataset)
            del train_df, x_train, y_train

            test_df = dataset[dataset.index.isin(test)]
            x_test, y_test = self.split_dataset(test_df)
            # Creating Tensorflow datasets
            atom_dim_test, bond_dim_test, cell_line_dim_test, test_dataset = self.tf_dataset_creator(x_test, y_test,
                                                                                                     batch_size,
                                                                                                     mpnn_dataset,
                                                                                                     conv_dataset)
            del test_df, x_test

            yield (atom_dim, bond_dim, cell_line_dim), train_dataset, test_dataset, y_test