# Python
import os
import shutil
# Downloaded
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
# SignDex
from signdex.processing import Processor
from signdex.common import Camera, Path
from signdex.dev import Window

class Dataset:
    def __init__(self, name):
        self.name = name
        self.path = os.path.join(Path.DATASETS, self.name)
        self.train_path = os.path.join(self.path,'training')
        self.test_path = os.path.join(self.path,'testing')
        self.camera = Camera()
        self.processor = Processor()

    def summary(self):
        print('DATASET:',self.name)

        if self.__exists():
            print('  Location:', self.path)
            print('  Total classes:', self.total_tags)
            print('  Class list:', self.tag_list)
            print('  Total images:', self.total_images)
        else:
            print('  Dataset does not exist')

    def create(self, tags, n, side='left', ratio=0.8, replace_if_exists=False):
        # Clean tag list from empty values
        tags = [s.strip() for s in tags if s.strip()]
        if len(tags) == 0:
            raise ValueError('tag list cannot be empty')

        # Check if dataset already exists
        if self.__exists():
            # If specified, delete dataset if exists
            if replace_if_exists:
                self.delete()
            else:
                raise Exception('dataset already exists')

        # Total images per set
        total_training_images = int(n*ratio)

        # Create train and test directories
        self.__create_train_test_dirs(tags)

        self.camera.open()
        cv2.namedWindow('video')

        for tag in tags:
            tag = tag.strip()
            tag_count = 0
            recording = False

            # Create directory
            train_tagpath = os.path.join(self.train_path, tag)
            test_tagpath = os.path.join(self.test_path, tag)

            # Wait for user to start recording
            while True:
                Window.show_tag_panel(tag, (0, n), recording)
                images = self.__get_processed_images(side)
                cv2.imshow('video', images['display'])
                if cv2.waitKey(1) & 0xFF == ord('c'): break
            recording = True

            # Start recording
            while True:
                images = self.__get_processed_images(side)
                cv2.imshow('video', images['display'])
                Window.show_tag_panel(tag, (tag_count, n), recording)

                # Save images to training or testing
                tagname = '{}{}.jpg'.format(tag,('0000'+str(tag_count+1))[-4:])
                if tag_count < total_training_images:
                    savepath = os.path.join(train_tagpath,tagname)
                    cv2.imwrite(savepath, images['crop'])
                    if tag_count != n: tag_count += 1
                elif tag_count < n:
                    savepath = os.path.join(test_tagpath,tagname)
                    cv2.imwrite(savepath, images['crop'])
                    if tag_count != n: tag_count += 1

                if cv2.waitKey(1) != -1: break

    def load(self, target_size, binarized=False):
        # Image Data Generator instances
        training_idg = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        testing_idg = ImageDataGenerator(rescale=1.0/255.)

        # Directory flows
        training_generator = training_idg.flow_from_directory(
            self.train_path,
            batch_size=32,
            class_mode='categorical',
            target_size=target_size
        )
        testing_generator = testing_idg.flow_from_directory(
            self.test_path,
            batch_size=32,
            class_mode='categorical',
            target_size=target_size
        )

        return {
            "training": training_generator,
            "testing": testing_generator
        }

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def __get_processed_images(self, side):
        frame = self.camera.read()
        image = self.processor.draw_square(frame, side)
        crop = self.processor.crop(frame, side)

        return {
            "frame":frame,
            "display":image,
            "crop":crop
        }

    def __create_train_test_dirs(self, tags):
        os.makedirs(self.train_path)
        os.makedirs(self.test_path)

        for tag in tags:
            os.mkdir(os.path.join(self.train_path, tag))
            os.mkdir(os.path.join(self.test_path, tag))

    def __exists(self):
        dataset_list = Path.list_subdirectories(Path.DATASETS)

        if self.name in dataset_list: return True
        else: return False
    
    @property
    def total_images(self):
        total = 0
        
        for tag in self.tag_list:
            total += len(Path.list_files(os.path.join(self.train_path, tag)))
            total += len(Path.list_files(os.path.join(self.test_path, tag)))
        
        return total

    @property
    def total_tags(self):
        return len(Path.list_subdirectories(self.train_path))
    
    @property
    def tag_list(self):
        return Path.list_subdirectories(self.train_path)