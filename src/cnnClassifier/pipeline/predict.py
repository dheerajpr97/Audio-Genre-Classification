import numpy as np
import os
import pandas as pd
import tensorflow as tf
import librosa
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        self.test_audio_features = []
        self.genre_dict = {'blues': 0,
                      'classical': 1,
                      'country': 2,
                      'disco': 3,
                      'hiphop': 4,
                      'jazz': 5,
                      'metal': 6,
                      'pop': 7,
                      'reggae': 8,
                      'rock': 9}
        
        model = tf.keras.models.load_model(os.path.join("artifacts","model_training","model.h5"))

        #test_audio = self.filename
        #load_test_audio = librosa.load(test_audio)
        signal, sr = librosa.load(self.filename, sr=22050)
        if len(signal) > 661500:
            signal = signal[0:661500]
        else:
            signal = np.pad(signal, (0, 661500 - signal.shape[0]), 'wrap')
                    
        chunk_duration = 3  # seconds
        chunk_samples = int(chunk_duration * sr)
        num_chunks = int(np.ceil(len(signal) / chunk_samples))
        audio_chunks = [signal[i*chunk_samples:(i+1)*chunk_samples] for i in range(num_chunks)]
        for chunk in audio_chunks:
            mfcc = librosa.feature.mfcc(y=chunk, n_fft=2048, hop_length=512, n_mfcc=13, sr=sr)
            mfcc = np.array(mfcc.T) # Transpose the matrix to get the shape (n, 13)
            
            self.test_audio_features.append(mfcc)

        self.test_audio_features = np.array(self.test_audio_features)
        self.test_audio_features = self.test_audio_features.reshape(
            -1, self.test_audio_features[0].shape[0], self.test_audio_features[0].shape[1], 1)    
        
        result = np.argmax(model.predict(self.test_audio_features), axis=1)
           
        result_idx = int(np.bincount(result).argmax())
        for key, value in self.genre_dict.items():
            if value == result_idx:
                return [{"audio": key}]
                                        