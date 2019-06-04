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
    def list_subdirectories(dirpath, return_fullpath=False):
        dirlist = []
        for file_ in os.listdir(dirpath):
            filepath = os.path.join(dirpath,file_)
            if os.path.isdir(filepath):
                if return_fullpath:
                    dirlist.append(filepath)
                else:
                    dirlist.append(file_)
        
        return dirlist

    @staticmethod
    def list_files(dirpath, return_fullpath=False):
        filelist = []
        for file_ in os.listdir(dirpath):
            filepath = os.path.join(dirpath,file_)
            if os.path.isfile(filepath):
                if return_fullpath:
                    filelist.append(filepath)
                else:
                    filelist.append(file_)
        
        return filelist