import os
import urllib.request as request
import zipfile
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, MaxPooling2D, Input, Activation, Dense
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.input_shape = self.config.params_image_size
        self.num_classes = self.config.params_classes
        self.model = self.prepare_model()

    #@staticmethod    
    def prepare_model(self):
        self.model = Sequential()
        self.model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=self.input_shape, kernel_initializer = 'he_normal'))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization())

        # 2nd conv layer
        self.model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer = 'he_normal'))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization())

        # 3rd conv layer
        self.model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer = 'he_normal'))
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization())

        # flatten output and feed it into dense layer
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu', kernel_initializer = 'he_normal'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(64, activation='relu', kernel_initializer = 'he_normal'))
        self.model.add(keras.layers.Dropout(0.5))
        # output layer
        self.model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        self.model.summary()
        #return self.model       

        self.save_model(path=self.config.model_path, model=self.model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)