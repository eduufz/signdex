import os

import cv2

class Camera:
    def __init__(self):
        pass
        
    def open(self, source=0):
        self.source = cv2.VideoCapture(source)
        
        if(self.source is None or not self.source.isOpened()):
            raise Warning('no camera found')

    def read(self):
        _, frame = self.source.read()
        
        return frame
    
    def close(self):
        self.source.release()
        cv2.destroyAllWindows()

class Path:
    DATASETS = os.path.join(os.getcwd(), 'datasets')

    @staticmethod
    def list_subdirectories(dirpath):
        return [d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,d))]