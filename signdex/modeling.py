# Python
import os
# Downloaded
import tensorflow as tf
# SignDex
from signdex.data import Dataset
from signdex.processing import Processor
from signdex.common import Path

class Model:
    def __init__(self, name):
        self.name = name
        self.path = os.path.join(Path.MODELS, self.name)
        self.__model = None
        self.load()

    def load(self):
        if self.__exists():
            # Read json model
            json_model_path = os.path.join(self.path, '{}.json'.format(self.name))
            json_file = open(json_model_path, 'r')
            json_model = json_file.read()
            json_file.close()

            # Load model from json
            self.__model = tf.keras.models.model_from_json(json_model)
    
    def __exists(self):
        model_list = Path.list_subdirectories(Path.MODELS)

        if self.name in model_list: return True
        else: return False