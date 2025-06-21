import logging
from functools import lru_cache
from helper.enum.dataset.data_type import DataType

from src.comet.use_comet_strategy import UseCometStrategy
from src.comet.skip_comet_strategy import SkipCometStrategy

from src.dataset.random_split_dataset_strategy import RandomSplitDatasetStrategy
from src.dataset.cell_stratified_dataset_strategy import CellStratifiedDatasetStrategy
from src.dataset.drug_stratified_dataset_strategy import DrugStratifiedDatasetStrategy
from src.dataset.drug_cell_stratified_dataset_strategy import DrugCellStratifiedDatasetStrategy
from src.dataset.cross_domain_dataset_strategy import CrossDomainDatasetStrategy

from src.model.training.stratified_split_model_training_strategy import StratifiedSplitTrainingStrategy
from src.model.training.random_split_training_strategy import RandomSplitTrainingStrategy

from src.model.build.classification_model_build_strategy import ClassificationModelCreationStrategy
from src.model.build.regression_model_build_strategy import RegressionModelCreationStrategy

from src.model.learning_task.classification_learning_task_strategy import ClassificationLearningTaskStrategy
from src.model.learning_task.regression_learning_task_strategy import RegressionLearningTaskStrategy

class StrategyCreator:
    def __init__(self, args):
        self.args = args

    @property
    def use_comet(self):
        return self.args.use_comet

    @property
    def data_source(self):
        return self.args.data_source

    @property
    def evaluation_source(self):
        return self.args.evaluation_source

    @property
    def data_type(self):
        return self.args.data_type

    @property
    def split_type(self):
        return self.args.split_type

    @property
    def random_state(self):
        return self.args.random_state

    @property
    def batch_size(self):
        return self.args.batch_size

    @property
    def epoch(self):
        return self.args.epoch

    @property
    def learning_rate(self):
        return self.args.learning_rate

    @property
    def learning_task(self):
        return self.args.learning_task

    @property
    def selformer_trainable_layers(self):
        return self.args.selformer_trainable_layers


    def get_comet_strategy(self):
        strategies = { True: UseCometStrategy(), False: SkipCometStrategy() }
        return strategies[self.use_comet]

    @lru_cache(maxsize=1)
    def get_dataset_path_by_data_type(self):
        """
        Get the dataset path based on data type and source.
        
        Returns:
            str: Full path to the dataset
            
        Raises:
            ValueError: If data_type is not recognized
        """
        type_path_map = {
            DataType.normal.label: DataType.normal.path,
            DataType.l1000.label: DataType.l1000.path,
            DataType.l1000_cross_domain.label: DataType.l1000_cross_domain.path,
            DataType.pathway.label: DataType.pathway.path,
            DataType.pathway_reduced.label: DataType.pathway_reduced.path,
            DataType.digestive.label: DataType.digestive.path
        }
        
        relative_path = type_path_map.get(self.data_type)
        if relative_path is None:
            valid_types = list(type_path_map.keys())
            raise ValueError(f"Unknown data_type: {self.data_type}. Valid options: {valid_types}")

        return DataType.prefix.path + self.data_source + relative_path

    def get_evaluation_dataset_path_by_data_type(self):
        """
        Get the evaluation dataset path for cross-domain scenarios.
        
        Returns:
            str: Full path to the evaluation dataset
            
        Raises:
            ValueError: If evaluation_source is None or data_type is invalid
        """
        if self.evaluation_source is None:
            raise ValueError("evaluation_source must be provided for cross_domain split type.")

        type_path_map = {
            DataType.normal.label: DataType.normal.path,
            DataType.l1000.label: DataType.l1000.path,
            DataType.l1000_cross_domain.label: DataType.l1000_cross_domain.path,
            DataType.pathway.label: DataType.pathway.path,
            DataType.pathway_reduced.label: DataType.pathway_reduced.path,
            DataType.digestive.label: DataType.digestive.path
        }
        
        relative_path = type_path_map.get(self.data_type)
        if relative_path is None:
            valid_types = list(type_path_map.keys())
            raise ValueError(f"Unknown data_type for evaluation: {self.data_type}. Valid options: {valid_types}")

        if self.data_type == DataType.l1000_cross_domain.label and self.evaluation_source != self.data_source:
             logging.info(f"Using evaluation_source for l1000_cross_domain: {self.evaluation_source}")
             return DataType.prefix.path + self.evaluation_source + relative_path

        return DataType.prefix.path + self.evaluation_source + relative_path

    def get_split_strategy(self):
        split_type = self.split_type

        if split_type == 'random':
            dataset_strategy = RandomSplitDatasetStrategy(self.get_dataset_path_by_data_type())
            training_strategy = RandomSplitTrainingStrategy()
        elif split_type == 'cell_stratified':
            dataset_strategy = CellStratifiedDatasetStrategy(self.get_dataset_path_by_data_type())
            training_strategy = StratifiedSplitTrainingStrategy()
        elif split_type == 'drug_stratified':
            dataset_strategy = DrugStratifiedDatasetStrategy(self.get_dataset_path_by_data_type())
            training_strategy = StratifiedSplitTrainingStrategy()
        elif split_type == 'drug_cell_stratified':
            dataset_strategy = DrugCellStratifiedDatasetStrategy(self.get_dataset_path_by_data_type())
            training_strategy = StratifiedSplitTrainingStrategy()
        elif split_type == 'cross_domain':
            dataset_strategy = CrossDomainDatasetStrategy(self.get_dataset_path_by_data_type(),
                                                          self.get_evaluation_dataset_path_by_data_type())
            training_strategy = RandomSplitTrainingStrategy()
        else:
            raise ValueError(f"Unknown split_type: {split_type}")

        return {'dataset': dataset_strategy, 'training': training_strategy}

    def get_model_creation_strategy(self):
        if self.learning_task == 'classification':
            return ClassificationModelCreationStrategy()
        elif self.learning_task == 'regression':
            return RegressionModelCreationStrategy()
        else:
            raise ValueError(f"Unknown learning_task: {self.learning_task}")

    def get_learning_task_strategy(self):
        if self.learning_task == 'classification':
            return ClassificationLearningTaskStrategy()
        elif self.learning_task == 'regression':
            return RegressionLearningTaskStrategy()
        else:
            raise ValueError(f"Unknown learning_task: {self.learning_task}")