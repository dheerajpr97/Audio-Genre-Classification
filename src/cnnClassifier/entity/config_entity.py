from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_path: Path
    local_data_file: Path
    unzip_dir: Path
    

@dataclass(frozen=True)
class PrepareDataConfig:
    root_dir: Path
    source_dir: Path
    target_dir: Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_classes: int
    

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
   

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_path: Path
    trained_model_path: Path
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path
    params_epochs: int
    params_batch_size: int
    params_shuffle: bool
    params_image_size: list