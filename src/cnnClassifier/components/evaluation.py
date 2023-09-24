from urllib.parse import urlparse
from cnnClassifier.utils.common import MFCCDataGenerator
import pandas as pd
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def get_test_data(self):
        self.test_df = pd.read_json(self.config.test_data_path)


    def test_generator(self):
        self.test_data_generator = MFCCDataGenerator(
            self.test_df, batch_size=32, shuffle=False)

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self.test_generator()
        self.score = self.model.evaluate(self.test_data_generator)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)  
