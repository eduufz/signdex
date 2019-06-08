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

        self.path_binarized = os.path.join(Path.DATASETS, '{}_binarized'.format(self.name))
        self.train_path_binarized = os.path.join(self.path_binarized,'training')
        self.test_path_binarized = os.path.join(self.path_binarized,'testing')

        self.camera = Camera()
        self.processor = Processor()

    def summary(self):
        print('DATASET:',self.name)

        if self.exists():
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

        # If specified, delete dataset if exists
        if replace_if_exists:
            self.delete()

        # Total images per set
        total_training_images = int(n*ratio)

        # Create train and test directories
        self.__create_train_test_dirs(tags)

        self.camera.open()
        cv2.namedWindow('video')

        for tag in tags:
            tag = tag.strip()
            tag_exist_count = self.total_tag_images(tag)
            tag_count = 0
            recording = False
            end = False

            # Create directory
            train_tagpath = os.path.join(self.train_path, tag)
            test_tagpath = os.path.join(self.test_path, tag)

            # Wait for user to start recording
            while not recording and not end:
                Window.show_tag_panel(tag, (0, n), recording)
                images = self.__get_processed_images(side)
                cv2.imshow('video', images['display'])

                key = cv2.waitKey(1)
                if key & 0xFF == ord('c'): recording = True
                elif key & 0xFF == 27: end = True

            # Start recording
            while recording and not end:
                images = self.__get_processed_images(side)
                cv2.imshow('video', images['display'])
                Window.show_tag_panel(tag, (tag_count, n), recording)

                # Save images to training or testing
                tagname = '{}{}.jpg'.format(tag,('0000'+str(tag_exist_count+tag_count+1))[-4:])
                if tag_count < total_training_images:
                    savepath = os.path.join(train_tagpath,tagname)
                    cv2.imwrite(savepath, images['crop'])
                    if tag_count != n: tag_count += 1
                elif tag_count < n:
                    savepath = os.path.join(test_tagpath,tagname)
                    cv2.imwrite(savepath, images['crop'])
                    if tag_count != n: tag_count += 1
                
                key = cv2.waitKey(1)
                if key != -1: recording = False
                    
            if end: break

    def process(self):
        if not self.exists():
            raise ValueError('dataset does not exist')
        
        self.__create_train_test_dirs(self.tag_list, binarized=True)

        train_tagpaths = Path.list_subdirectories(self.train_path, return_fullpath=True)
        train_bin_tagpaths = Path.list_subdirectories(self.train_path_binarized, return_fullpath=True)
        test_tagpaths = Path.list_subdirectories(self.test_path, return_fullpath=True)
        test_bin_tagpaths = Path.list_subdirectories(self.test_path_binarized, return_fullpath=True)

        print('Processing dataset...')

        for tagpath,bin_tagpath in zip(train_tagpaths,train_bin_tagpaths):
            file_list = Path.list_files(tagpath, return_fullpath=True)
            for filepath in file_list:
                filename = filepath.replace('\\','/').split('/')[-1]
                image = cv2.imread(filepath)
                image = self.processor.binarize(image)
                cv2.imshow('processed',image)
                cv2.waitKey(1)
                cv2.imwrite(os.path.join(bin_tagpath,filename), image)
        
        for tagpath,bin_tagpath in zip(test_tagpaths,test_bin_tagpaths):
            file_list = Path.list_files(tagpath, return_fullpath=True)
            for filepath in file_list:
                filename = filepath.replace('\\','/').split('/')[-1]
                image = cv2.imread(filepath)
                image = self.processor.binarize(image)
                cv2.imshow('processed',image)
                cv2.waitKey(1)
                cv2.imwrite(os.path.join(bin_tagpath,filename), image)
        cv2.destroyAllWindows()

        print('{} total images processed'.format(self.total_images))

    def load(self, target_size, binarized=False):
        color_mode = ('grayscale' if binarized else 'rgb')
        train_path = (self.train_path_binarized if binarized else self.train_path)
        test_path = (self.test_path_binarized if binarized else self.test_path)

        # Image Data Generator instances
        training_idg = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        testing_idg = ImageDataGenerator(rescale=1.0/255.)

        # Directory flows
        training_generator = training_idg.flow_from_directory(
            train_path,
            batch_size=32,
            class_mode='categorical',
            target_size=target_size,
            color_mode=color_mode
        )
        testing_generator = testing_idg.flow_from_directory(
            test_path,
            batch_size=32,
            class_mode='categorical',
            target_size=target_size,
            color_mode=color_mode
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

    def __create_train_test_dirs(self, tags, binarized=False):
        train_path = (self.train_path_binarized if binarized else self.train_path)
        test_path = (self.test_path_binarized if binarized else self.test_path)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for tag in tags:
            os.makedirs(os.path.join(train_path, tag), exist_ok=True)
            os.makedirs(os.path.join(test_path, tag), exist_ok=True)

    def exists(self):
        dataset_list = Path.list_subdirectories(Path.DATASETS)

        if self.name in dataset_list: return True
        else: return False
    
    def total_tag_images(self, tag):
        total = 0

        if tag in self.tag_list:
            total += len(Path.list_files(os.path.join(self.train_path, tag)))
            total += len(Path.list_files(os.path.join(self.test_path, tag)))

        return total
    
    @property
    def total_images(self):
        total = 0
        
        for tag in self.tag_list:
            total += len(Path.list_files(os.path.join(self.train_path, tag)))
            total += len(Path.list_files(os.path.join(self.test_path, tag)))
        
        return total

    @property
    def total_tags(self):
        return len(self.tag_list)
    
    @property
    def tag_list(self):
        return Path.list_subdirectories(self.train_path)