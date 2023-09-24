import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def decodeAudio(audiostring, fileName):
    audiodata = base64.b64decode(audiostring)
    with open(fileName, 'wb') as f:
        f.write(audiodata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    

class MFCCDataGenerator(tf.keras.utils.Sequence): 
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.dataframe = dataframe # dataframe with the mfcc features and the genre
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.num_classes = len(dataframe['genre'].unique()) # number of classes
        
        self.label_encoder = LabelEncoder() 
        self.label_encoder.fit(dataframe['genre']) # encode the genre labels
        
        self.scaler = StandardScaler() # standardize the mfcc features
        self.scaler.fit(np.concatenate(dataframe['MFCC features']))
        
        self.indexes = np.arange(len(dataframe))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size)) # number of batches
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_data = self.dataframe.iloc[batch_indexes]
        
        X, y = self.__data_generation(batch_data)
        return X, y
    
    def __data_generation(self, batch_data):
        mfcc_features = batch_data['MFCC features'].apply(lambda x: self.scaler.transform(x)).tolist() # standardize the mfcc features
        mfcc_features = np.array(mfcc_features).reshape(-1, mfcc_features[0].shape[0], mfcc_features[0].shape[1], 1) 
       
        genre_labels = self.label_encoder.transform(batch_data['genre'])
        genre_one_hot =  to_categorical(genre_labels, num_classes=self.num_classes)
        
        return mfcc_features, genre_one_hot
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
