# Python
import os
# Downloaded
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
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

        self.__train(model, dataset.load(target_size=target_size, binarized=True))

    def create_cnn(self, dataset, target_size=(64,64)):
        input_shape = (target_size[0], target_size[1], 3)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
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

        self.__train(model, dataset.load(target_size=target_size, binarized=False))

    def predict(self, image):
        size = int(self.name.split('_')[-1])
        image = cv2.resize(image, (size,size))

        results = self.__model.predict([[image]])

        return results
    
    def __train(self, model, generators):
        history = model.fit_generator(
            generators['training'],
            epochs=15,
            verbose=1,
            validation_data=generators['testing']
        )

        self.__save(model)
        self.__graph_results(history)

    def __save(self, model):
        os.makedirs(self.path)

        # Save JSON model
        with open(self.model_path, 'w') as json_file:
            json_file.write(model.to_json())

        # Save model weights
        model.save_weights(self.weights_path)
    
    def __graph_results(self, history):
        #-----------------------------------------------------------
        # Retrieve a list of list results on training and test data
        # sets for each training epoch
        #-----------------------------------------------------------
        acc=history.history['acc']
        val_acc=history.history['val_acc']
        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs=range(len(acc)) # Get number of epochs

        #------------------------------------------------
        # Plot training and validation accuracy per epoch
        #------------------------------------------------
        plt.plot(epochs, acc, 'r', "Training Accuracy")
        plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
        plt.title('Training and validation accuracy')
        plt.figure()

        #------------------------------------------------
        # Plot training and validation loss per epoch
        #------------------------------------------------
        plt.plot(epochs, loss, 'r', "Training Loss")
        plt.plot(epochs, val_loss, 'b', "Validation Loss")
        plt.figure()
    
    def __exists(self):
        model_list = Path.list_subdirectories(Path.MODELS)

        if self.name in model_list: return True
        else: return False