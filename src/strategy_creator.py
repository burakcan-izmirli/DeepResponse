""" Strategy creator """
import logging
from helper.enum.dataset.data_type import DataType

from src.comet.use_comet_strategy import UseCometStrategy
from src.comet.skip_comet_strategy import SkipCometStrategy

from src.dataset.random_split_dataset_strategy import RandomSplitDatasetStrategy
from src.dataset.cell_stratified_dataset_strategy import CellStratifiedDatasetStrategy
from src.dataset.drug_stratified_dataset_strategy import DrugStratifiedDatasetStrategy
from src.dataset.drug_cell_stratified_dataset_strategy import DrugCellStratifiedDatasetStrategy
from src.dataset.cross_domain_dataset_strategy import CrossDomainDatasetStrategy

from src.model.build.merged_model_build_strategy import MergedModelStrategy

from src.model.training.stratified_split_model_training_strategy import StratifiedSplitTrainingStrategy
from src.model.training.random_split_training_strategy import RandomSplitTrainingStrategy


class StrategyCreator:
    """ Strategy creator """

    def __init__(self, use_comet, data_source, evaluation_source, data_type, split_type, random_state, batch_size,
                 epoch, learning_rate):
        self.use_comet = use_comet
        self.data_source = data_source
        self.evaluation_source = evaluation_source
        self.data_type = data_type
        self.split_type = split_type
        self.random_state = random_state
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        logging.basicConfig(level=logging.INFO)

    def get_comet_strategy(self):
        """ Get comet strategy """
        strategies = {
            True: UseCometStrategy(),
            False: SkipCometStrategy()
        }
        return strategies[self.use_comet]

    def get_dataset_path_by_data_type(self):
        """ Get dataset path by data type """
        paths = {
            DataType.normal.label: DataType.prefix.path + self.data_source + DataType.normal.path,
            DataType.l1000.label: DataType.prefix.path + self.data_source + DataType.l1000.path,
            DataType.l1000_cross_domain.label: DataType.prefix.path + self.data_source + DataType.l1000_cross_domain.path,
            DataType.pathway.label: DataType.prefix.path + self.data_source + DataType.pathway.path,
            DataType.pathway_reduced.label: DataType.prefix.path + self.data_source + DataType.pathway_reduced.path,
            DataType.digestive.label: DataType.prefix.path + self.data_source + DataType.digestive.path
        }
        return paths[self.data_type]

    def get_evaluation_dataset_path_by_data_type(self):
        """ Get evaluation dataset path by data type """
        paths = {
            DataType.normal.label: DataType.prefix.path + self.evaluation_source + DataType.normal.path,
            DataType.l1000.label: DataType.prefix.path + self.evaluation_source + DataType.l1000.path,
            DataType.l1000_cross_domain.label: DataType.prefix.path + self.data_source + DataType.l1000_cross_domain.path,
            DataType.pathway.label: DataType.prefix.path + self.evaluation_source + DataType.pathway.path,
            DataType.pathway_reduced.label: DataType.prefix.path + self.evaluation_source + DataType.pathway_reduced.path,
            DataType.digestive.label: DataType.prefix.path + self.evaluation_source + DataType.digestive.path
        }
        return paths[self.data_type]

    def get_split_strategy(self):
        """ Get split strategy """
        strategies = {
            'random': {'dataset': RandomSplitDatasetStrategy(self.get_dataset_path_by_data_type()),
                       'training': RandomSplitTrainingStrategy()},
            'cell_stratified': {'dataset': CellStratifiedDatasetStrategy(self.get_dataset_path_by_data_type()),
                                'training': StratifiedSplitTrainingStrategy()},
            'drug_stratified': {'dataset': DrugStratifiedDatasetStrategy(self.get_dataset_path_by_data_type()),
                                'training': StratifiedSplitTrainingStrategy()},
            'drug_cell_stratified': {'dataset': DrugCellStratifiedDatasetStrategy(self.get_dataset_path_by_data_type()),
                                     'training': StratifiedSplitTrainingStrategy()},
            'cross_domain': {'dataset': CrossDomainDatasetStrategy(self.get_dataset_path_by_data_type(),
                                                                   self.get_evaluation_dataset_path_by_data_type()),
                             'training': RandomSplitTrainingStrategy()},
        }
        return strategies[self.split_type]

    def get_model_creation_strategy(self):
        """ Get model strategy """
        strategies = {
            'classification': ClassificationModelCreationStrategy(),
            'regression': RegressionModelCreationStrategy(),
        }
        return strategies[self.learning_task]
    
    def get_learning_task_strategy(self):
        """ Get learning task strategy """
        strategies = {
            'classification': ClassificationLearningTaskStrategy(),
            'regression': RegressionLearningTaskStrategy(),
        }
        return strategies[self.learning_task]
