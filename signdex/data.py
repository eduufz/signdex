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
        paths = self.__create_train_test_dirs(tags)

        self.camera.open()
        cv2.namedWindow('video')

        for tag in tags:
            tag = tag.strip()
            tag_count = 0
            recording = False

            # Create directory
            train_tagpath = os.path.join(paths['training'], tag)
            test_tagpath = os.path.join(paths['testing'], tag)

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
        pass

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
        training_path = os.path.join(self.path,'training')
        testing_path = os.path.join(self.path,'testing')

        os.makedirs(training_path)
        os.makedirs(testing_path)

        for tag in tags:
            os.mkdir(os.path.join(training_path,tag))
            os.mkdir(os.path.join(testing_path,tag))

        return {"training":training_path, "testing":testing_path}

    def __exists(self):
        dataset_list = Path.list_subdirectories(Path.DATASETS)

        if self.name in dataset_list: return True
        else: return False
    
    @property
    def total_images(self):
        total = 0
        tag_list = Path.list_subdirectories(self.path)
        
        for tag in tag_list:
            total += len(Path.list_files(os.path.join(self.path, tag)))
        
        return total

    @property
    def total_tags(self):
        return len(Path.list_subdirectories(self.path))
    
    @property
    def tag_list(self):
        return Path.list_subdirectories(self.path)