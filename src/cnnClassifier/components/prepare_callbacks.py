import os
import tensorflow as tf
from tensorflow import keras
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            
        )
    
    @property
    def _create_early_stopping_callbacks(self):
        return keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            verbose=1,
            restore_best_weights=True,
        )
    
    @property
    def _create_reduce_lr_on_plateau_callbacks(self):
        return keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=15,
            verbose=1,
            min_lr=1e-4,
        )


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._create_early_stopping_callbacks,
            self._create_reduce_lr_on_plateau_callbacks
            ]