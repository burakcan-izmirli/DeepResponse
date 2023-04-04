""" Drug cell stratified dataset strategy """
import math
import numpy as np

from helper.enum.dataset.n_split import NSplit

from src.dataset.base_dataset_strategy import BaseDatasetStrategy


class DrugCellStratifiedDatasetStrategy(BaseDatasetStrategy):
    """ Drug cell stratified dataset strategy """

    def create_splitter(self, dataset, random_state):
        """ Create splitter """
        pass

    def split_dataset(self, dataset, *args, **kwargs):
        """ Split dataset """

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

    def prepare_dataset(self, dataset, split_type, batch_size, random_state):
        """ Prepare dataset """
        pass
