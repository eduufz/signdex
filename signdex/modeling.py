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
        self.model_path = os.path.join(self.path, '{}.json'.format(self.name))
        self.weights_path = os.path.join(self.path, '{}.h5'.format(self.name))
        self.__model = None
        self.load()

    def load(self):
        if self.__exists():
            # Read json model
            json_file = open(self.model_path, 'r')
            json_model = json_file.read()
            json_file.close()

            # Load model from json
            self.__model = tf.keras.models.model_from_json(json_model)

    def create_nn(self, dataset, target_size=(64,64)):
        input_size = target_size[0]*target_size[1]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_size, activation='relu', input_shape=(input_size,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(dataset.total_tags, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        self.__train(model, dataset.load())

    def create_cnn(self, dataset, target_size=(64,64)):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(dataset.total_tags, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        self.__train(model, dataset.load())
    
    def __train(self, model, generator):
        pass

    def __save(self, model):
        # Save JSON model
        with open(self.model_path, 'w') as json_file:
            json_file.write(model.to_json())

        # Save model weights
        model.save_weights(self.weights_path)
    
    def __exists(self):
        model_list = Path.list_subdirectories(Path.MODELS)

        if self.name in model_list: return True
        else: return False