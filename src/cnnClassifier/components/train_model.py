import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils.common import MFCCDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_model(self):
        self.model = tf.keras.models.load_model(
            self.config.model_path
        )

    def get_datasets(self):
        self.train_df = pd.read_json(self.config.train_data_path)
        self.val_df = pd.read_json(self.config.val_data_path)
        self.test_df = pd.read_json(self.config.test_data_path)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):
        self.train_data_generator = MFCCDataGenerator(self.train_df, batch_size=64, shuffle=True)
        self.val_data_generator = MFCCDataGenerator(self.val_df, batch_size=64, shuffle=False)

        #self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        #self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.history = self.model.fit(
            self.train_data_generator,
            epochs=self.config.params_epochs,
            #steps_per_epoch=self.steps_per_epoch,
            #validation_steps=self.validation_steps,
            validation_data=self.val_data_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )