import os
from cnnClassifier.constant import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareDataConfig,
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig
                                                )

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
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            val_data_path=config.val_data_path
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
    
    def get_training_config(self) -> TrainingConfig:
        model_training = self.config.model_training
        prepare_base_model = self.config.prepare_base_model
        prepare_data = self.config.prepare_data
        params = self.params
        create_directories([
            Path(model_training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(model_training.root_dir),
            model_path=Path(prepare_base_model.model_path),
            trained_model_path=Path(model_training.trained_model_path),
            train_data_path=Path(prepare_data.train_data_path), 
            test_data_path=Path(prepare_data.test_data_path),
            val_data_path=Path(prepare_data.val_data_path),     
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_shuffle=params.SHUFFLE,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config