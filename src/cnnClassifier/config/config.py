import os
from cnnClassifier.constant import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareDataConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_PATH,
        params_filepath = PARAMS_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_path=config.source_path,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_prepare_data_config(self) -> PrepareDataConfig:
        config = self.config.prepare_data

        create_directories([config.root_dir])

        prepare_data_config = PrepareDataConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            target_dir=config.target_dir,
        )

        return prepare_data_config
    
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=config.checkpoint_model_filepath
        )

        return prepare_callback_config