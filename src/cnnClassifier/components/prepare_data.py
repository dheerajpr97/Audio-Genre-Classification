import os
import pandas as pd
import librosa
import numpy as np
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareDataConfig
from pathlib import Path
from sklearn.model_selection import train_test_split

class PrepareData:
    def __init__(self, config: PrepareDataConfig):
        self.config = config       
        
    def dataframe_create(self):
        self.music_dataset = [] # File locations for each .wav file 
        self.genre_list = [] # Different Genres (or classes)
        self.music_path = self.config.source_dir
        
        for root, dirs, files in os.walk(self.music_path):
            for name in files:
                filename = os.path.join(root, name)
                self.music_dataset.append(filename)
                self.genre_list.append(filename.split("/")[3])
            self.music_df = pd.DataFrame({
            'file': self.music_dataset,
            'genre': self.genre_list
            })
        return self.music_df   
    
    def mfcc_feature_extraction(self, df):        
        self.X = []
        self.y = []
        self.y_cat = []
        self.mfcc = []
        self.mfcc_df = pd.DataFrame()

        for index, row in df.iterrows():
            signal, sr = librosa.load(row['file'], sr=22050)
            if len(signal) > 661500:
                signal = signal[0:661500]
            else:
                signal = np.pad(signal, (0, 661500 - signal.shape[0]), 'wrap')
                
            # Calculate the duration of each chunk
            chunk_duration = 5  # seconds
            # Calculate the number of samples in each chunk
            chunk_samples = int(chunk_duration * sr)
            # Calculate the total number of chunks
            num_chunks = int(np.ceil(len(signal) / chunk_samples))
            # Split the audio clip into chunks
            num_chunks = int(np.ceil(len(signal) / chunk_samples))
            audio_chunks = [signal[i*chunk_samples:(i+1)*chunk_samples] for i in range(num_chunks)]
            for chunk in audio_chunks:
                mfcc = librosa.feature.mfcc(y=chunk, n_fft=2048, hop_length=512, n_mfcc=13, sr=sr)
                mfcc = np.array(mfcc.T) # Transpose the matrix to get the shape (n, 13)
                
                self.X.append(mfcc)
                self.y.append(row['genre'])

        self.mfcc_df['MFCC features'] = self.X
        self.mfcc_df['genre'] = self.y
        
        return self.mfcc_df    

    def save_data(self, df):
        df.to_json(self.config.target_dir, index=False)
        logger.info(f"Data saved at {self.config.target_dir} in JSON format")          

    def train_val_test_split(self, df):
        self.train_df, self.test_df = train_test_split(
            df,
            test_size=0.15,
            random_state=42
        )
        self.train_df, self.val_df = train_test_split(
            self.train_df,
            test_size=0.2,
            random_state=42
        )
        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)
    
        self.train_df.to_json(self.config.train_data_path)
        self.val_df.to_json(self.config.val_data_path)
        self.test_df.to_json(self.config.test_data_path)
        logger.info(f"Train, Val, and Test datasets saved at {self.config.target_dir} in JSON format")      

