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

    def info(self):
        print('--- Dataset :',self.name, '---')
        print('Location:', self.path)
        print('Total images:')

    def create(self, tags, n):
        if self.__exists():
            raise Exception('dataset already exists')

        self.camera.open()
        cv2.namedWindow('video')

        for tag in tags:
            tag_count = 0
            started = False

            # Create directory
            tagpath = os.path.join(self.path, tag)
            os.makedirs(tagpath)

            # Wait for user to start recording
            while True:
                Window.show_tag_panel(tag, (0, n), started)
                
                frame = self.camera.read()
                cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('c'): break
            started = True

            # Start recording
            while True:
                Window.show_tag_panel(tag, (tag_count+1, n), started)

                frame = self.camera.read()
                cv2.imshow('video', frame)

                # Save images
                savepath = os.path.join(tagpath, '{}{}.jpg'.format(tag,tag_count+1))
                cv2.imwrite(savepath, frame)

                if cv2.waitKey(1) != -1: break

                if tag_count < n-1: tag_count += 1

    def load(self):
        pass

    def delete(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def __generate_report(self):
        pass

    def __exists(self):
        dataset_list = Path.list_subdirectories(Path.DATASETS)

        if self.name in dataset_list: return True
        else: return False