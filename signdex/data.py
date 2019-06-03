# Python
import os
# Downloaded
import cv2
# SignDex
from signdex.common import Camera, Path
from signdex.dev import Window

class Dataset:
    def __init__(self, name):
        self.camera = Camera()
        self.name = name

    def info(self):
        pass

    def create(self, tags, n):
        if self.__exists():
            raise Exception('dataset already exists')

        self.camera.open()
        cv2.namedWindow('video')

        for tag in tags:
            tag_count = 0
            started = False

            while True:
                Window.show_tag_panel(tag, (0, n), started)
                
                frame = self.camera.read()
                cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('c'): break
            started = True

            while True:
                Window.show_tag_panel(tag, (tag_count+1, n), started)

                frame = self.camera.read()
                cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('c'): break

                if tag_count < n-1: tag_count += 1


    def load(self):
        pass

    def delete(self):
        pass

    def __generate_report(self):
        pass

    def __exists(self):
        dataset_list = Path.list_subdirectories(Path.DATASETS)

        if self.name in dataset_list:
            return True
        else:
            return False