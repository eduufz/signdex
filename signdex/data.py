# Python
import os
import shutil
# Downloaded
import cv2
# SignDex
from signdex.common import Camera, Path
from signdex.dev import Window

class Dataset:
    def __init__(self, name):
        self.name = name
        self.path = os.path.join(Path.DATASETS, self.name)
        self.camera = Camera()

    def summary(self):
        print('DATASET:',self.name)

        if self.__exists():
            print('  Location:', self.path)
            print('  Total classes:', self.total_tags)
            print('  Class list:', self.tag_list)
            print('  Total images:', self.total_images)
        else:
            print('  Dataset does not exist')

    def create(self, tags, n, size=(200,200), replace_if_exists=False):
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

        self.camera.open()
        cv2.namedWindow('video')

        for tag in tags:
            tag = tag.strip()
            tag_count = 0
            recording = False

            # Create directory
            tagpath = os.path.join(self.path, tag)
            os.makedirs(tagpath)

            # Wait for user to start recording
            while True:
                Window.show_tag_panel(tag, (0, n), recording)
                
                frame = self.camera.read()
                cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('c'): break
            recording = True

            # Start recording
            while True:
                frame = self.camera.read()
                cv2.imshow('video', frame)
                Window.show_tag_panel(tag, (tag_count, n), recording)

                # Save images until specified
                if tag_count < n:
                    tagname = '{}{}.jpg'.format(tag,('0000'+str(tag_count+1))[-4:])
                    savepath = os.path.join(tagpath,tagname)
                    cv2.imwrite(savepath, frame)
                    if tag_count != n: tag_count += 1

                if cv2.waitKey(1) != -1: break

    def load(self):
        pass

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

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