"""
Configuration management for DeepResponse.

This module provides centralized configuration validation and management
for all DeepResponse components.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeepResponseConfig:
    """
    Configuration container for DeepResponse parameters.
    
    This class validates and stores all configuration parameters
    with proper type checking and constraint validation.
    """
    # Task configuration
    learning_task: str
    data_source: str
    evaluation_source: Optional[str]
    data_type: str
    split_type: str
    
    # Training parameters
    learning_rate: float
    batch_size: int
    epoch: int
    random_state: int
    
    # Model parameters
    selformer_trainable_layers: int
    
    # Monitoring
    use_comet: bool
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Comprehensive configuration validation."""
        self._validate_learning_task()
        self._validate_data_configuration()
        self._validate_training_parameters()
        self._validate_model_parameters()
        self._validate_cross_domain_consistency()
    
    def _validate_learning_task(self):
        """Validate learning task configuration."""
        valid_tasks = ['classification', 'regression']
        if self.learning_task not in valid_tasks:
            raise ValueError(f"learning_task must be one of {valid_tasks}, got: {self.learning_task}")
    
    def _validate_data_configuration(self):
        """Validate data source and type configuration."""
        valid_sources = ['depmap', 'gdsc', 'ccle', 'nci_60']
        valid_types = ['normal', 'l1000', 'l1000_cross_domain', 'pathway', 'pathway_reduced', 'digestive']
        
        if self.data_source not in valid_sources:
            raise ValueError(f"data_source must be one of {valid_sources}, got: {self.data_source}")
        
        if self.evaluation_source and self.evaluation_source not in valid_sources:
            raise ValueError(f"evaluation_source must be one of {valid_sources}, got: {self.evaluation_source}")
        
        if self.data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}, got: {self.data_type}")
        
        # Validate data source compatibility with data types
        gdsc_only_types = ['pathway', 'pathway_reduced', 'digestive']
        if self.data_type in gdsc_only_types and self.data_source != 'gdsc':
            raise ValueError(f"data_type '{self.data_type}' is only supported with data_source 'gdsc'")
    
    def _validate_training_parameters(self):
        """Validate training hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got: {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got: {self.batch_size}")
        
        if self.epoch <= 0:
            raise ValueError(f"epoch must be positive, got: {self.epoch}")
        
        if self.random_state < 0:
            raise ValueError(f"random_state must be non-negative, got: {self.random_state}")
    
    def _validate_model_parameters(self):
        """Validate model-specific parameters."""
        if self.selformer_trainable_layers < -1:
            raise ValueError(f"selformer_trainable_layers must be >= -1, got: {self.selformer_trainable_layers}")
    
    def _validate_cross_domain_consistency(self):
        """Validate cross-domain configuration consistency."""
        valid_splits = ['random', 'cell_stratified', 'drug_stratified', 'drug_cell_stratified', 'cross_domain']
        
        if self.split_type not in valid_splits:
            raise ValueError(f"split_type must be one of {valid_splits}, got: {self.split_type}")
        
        if self.split_type == 'cross_domain':
            if self.evaluation_source is None:
                raise ValueError("evaluation_source is required when split_type is 'cross_domain'")
            
            if self.evaluation_source == self.data_source:
                raise ValueError("evaluation_source must be different from data_source for cross_domain split")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'learning_task': self.learning_task,
            'data_source': self.data_source,
            'evaluation_source': self.evaluation_source,
            'data_type': self.data_type,
            'split_type': self.split_type,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'random_state': self.random_state,
            'selformer_trainable_layers': self.selformer_trainable_layers,
            'use_comet': self.use_comet
        }
    
    def get_summary(self) -> str:
        """Get a formatted configuration summary."""
        return f"""
DeepResponse Configuration Summary:
=====================================
Task Configuration:
  - Learning Task: {self.learning_task}
  - Data Source: {self.data_source}
  - Evaluation Source: {self.evaluation_source or 'None'}
  - Data Type: {self.data_type}
  - Split Type: {self.split_type}

Training Parameters:
  - Learning Rate: {self.learning_rate}
  - Batch Size: {self.batch_size}
  - Epochs: {self.epoch}
  - Random State: {self.random_state}

Model Configuration:
  - SELFormer Trainable Layers: {self.selformer_trainable_layers}

Monitoring:
  - Use Comet ML: {self.use_comet}
====================================="""


class ConfigurationManager:
    """Centralized configuration management for DeepResponse."""
    
    @staticmethod
    def from_args(args) -> DeepResponseConfig:
        """
        Create configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            DeepResponseConfig: Validated configuration object
        """
        return DeepResponseConfig(
            learning_task=args.learning_task,
            data_source=args.data_source,
            evaluation_source=args.evaluation_source,
            data_type=args.data_type,
            split_type=args.split_type,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epoch=args.epoch,
            random_state=args.random_state,
            selformer_trainable_layers=args.selformer_trainable_layers,
            use_comet=args.use_comet
        )
    
    @staticmethod
    def validate_data_availability(config: DeepResponseConfig) -> bool:
        """
        Validate that required datasets are available.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if all required datasets are available
            
        Raises:
            FileNotFoundError: If required datasets are missing
        """
        from helper.enum.dataset.data_type import DataType
        import os
        
        # Build the main dataset path
        type_path_map = {
            'normal': '/processed/',
            'l1000': '/processed/l1000/',
            'l1000_cross_domain': '/processed/l1000_cross_domain/',
            'pathway': '/processed/pathway/',
            'pathway_reduced': '/processed/pathway_reduced/',
            'digestive': '/processed/digestive/'
        }
        
        base_path = f"./dataset/{config.data_source}{type_path_map[config.data_type]}"
        
        # Check if main dataset directory exists
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Main dataset directory not found: {base_path}")
        
        # Check for essential files (adjust based on your actual file structure)
        essential_files = ['cell_features.csv', 'drug_features.csv', 'response_data.csv']
        missing_files = []
        
        for file_name in essential_files:
            file_path = os.path.join(base_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"Missing essential dataset files: {missing_files}")
        
        # Validate evaluation dataset if needed
        if config.evaluation_source:
            eval_path = f"./dataset/{config.evaluation_source}{type_path_map[config.data_type]}"
            if not os.path.exists(eval_path):
                raise FileNotFoundError(f"Evaluation dataset directory not found: {eval_path}")
            
            for file_name in essential_files:
                file_path = os.path.join(eval_path, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                raise FileNotFoundError(f"Missing evaluation dataset files: {missing_files}")
        
        logging.info(f"✓ Dataset validation passed: {config.data_source}/{config.data_type}")
        if config.evaluation_source:
            logging.info(f"✓ Evaluation dataset validated: {config.evaluation_source}/{config.data_type}")
        
        return True
