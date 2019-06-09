# Python
import os
import shutil
# Downloaded
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# SignDex
from signdex.data import Dataset
from signdex.processing import Processor
from signdex.common import Path


class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, limit):
        self.limit = limit
    
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > self.limit):
            print("\nAcurracy of {} achieved.".format(self.limit))
            self.model.stop_training = True

class Model:
    def __init__(self, name):
        self.name = name
        self.path = os.path.join(Path.MODELS, self.name)
        self.model_path = os.path.join(self.path, '{}.json'.format(self.name))
        self.weights_path = os.path.join(self.path, '{}.h5'.format(self.name))
        self.__model = None
        self.load()

    def load(self):
        if self.exists():
            # Read json model
            json_file = open(self.model_path, 'r')
            json_model = json_file.read()
            json_file.close()

            # Load model from json
            self.__model = tf.keras.models.model_from_json(json_model)
            self.__model.load_weights(self.weights_path)

            ds_name = '_'.join(self.name.split('_')[:-1])
            self.dataset = Dataset(ds_name)

    def create(self, dataset, target_size=(64,64), binarized=False, callback=-1):
        callbacks = []
        if callback != -1:
            callbacks.append(AccuracyCallback(callback))

        input_shape = (target_size[0], target_size[1], (1 if binarized else 3))

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(dataset.total_tags, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        self.__train(
            model,
            dataset.load(target_size=target_size, binarized=binarized),
            callbacks
        )

    def predict(self, image):
        size = int(self.name.split('_')[-1])
        image = cv2.resize(image, (size,size))

        image = [image.reshape(1,size,size,(1 if len(image.shape)==2 else 3))]
        tag = self.__model.predict_classes(image)[0]
        prediction = self.dataset.tag_list[tag]

        return prediction
    
    def __train(self, model, generators, callbacks=[]):
        history = model.fit_generator(
            generators['training'],
            steps_per_epoch=2000,
            epochs=50,
            verbose=1,
            validation_data=generators['testing'],
            validation_steps=800,
            callbacks=callbacks
        )

        self.__save(model)
        self.__graph_results(history)

    def __save(self, model):
        if self.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        os.makedirs(self.path)

        # Save JSON model
        with open(self.model_path, 'w') as json_file:
            json_file.write(model.to_json())

        # Save model weights
        model.save_weights(self.weights_path)

        print('Model {} saved to disk.'.format(self.name))
    
    def __graph_results(self, history):
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'],'r',linewidth=3.0)
        plt.plot(history.history['val_loss'],'b',linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves',fontsize=16)
        plt.show()

        plt.figure(figsize=[8,6])
        plt.plot(history.history['acc'],'r',linewidth=3.0)
        plt.plot(history.history['val_acc'],'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves',fontsize=16)
        plt.show()
    
    def exists(self):
        model_list = Path.list_subdirectories(Path.MODELS)

        if self.name in model_list: return True
        else: return False