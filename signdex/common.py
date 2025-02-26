# Python
import os
# Downloaded
import cv2
# SignDex


class Camera:
    def __init__(self):
        self.source = None
        
    def open(self, source=0):
        self.source = cv2.VideoCapture(source)
        
        if(not self.isavailable()):
            raise Warning('no camera found')

    def capture(self):
        _, frame = self.source.read()
        
        return cv2.flip(frame,1)
    
    def close(self):
        self.source.release()
        cv2.destroyAllWindows()
    
    def isavailable(self):
        return (self.source is not None and self.source.isOpened())

class Path:
    DATASETS = os.path.join(os.getcwd(), 'datasets')
    MODELS = os.path.join(os.getcwd(), 'models')

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