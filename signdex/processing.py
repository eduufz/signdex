# Python
import os
# Downloaded
import cv2
import numpy as np
# SignDex


class Processor:
    def __init__(self):
        self.params = {
            "position": {
                "left": (50,100),
                "right": (340,100)
            },
            "dimensions": (250,250),
            "threshold": {
                0: cv2.THRESH_BINARY+cv2.THRESH_OTSU,
                1: cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
            }
        }
    
    def binarize(self, image, method=0):
        if method not in self.params['threshold']:
            raise ValueError('only methods "0" and "1" are available')

        image = image.copy()
        
        # Set blue and green channels to 0
        image[:, :, 0] = 0
        image[:, :, 1] = 0
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur image
        img_blur = cv2.GaussianBlur(img_gray, (7,7), 0)
        
        # Threshold image
        _, img_thresh = cv2.threshold(img_blur, 125, 255, self.params['threshold'][method])
        
        return img_thresh
    
    def crop(self, image, side, resize=None):
        x,y = 0,0
        w,h = self.params['dimensions']

        if side in self.params['position']:
            x,y = self.params['position'][side]
        else:
            raise ValueError('only "right" and "left" side available')

        image = image[y:y+h, x:x+w]

        return image

    def draw_square(self, image, side, scolor=(0,255,0), thickness=0):
        image = image.copy()
        x,y = 0,0
        w,h = self.params['dimensions']

        if side in self.params['position']:
            x,y = self.params['position'][side]
        else:
            raise ValueError('only "right" and "left" side available')

        image = cv2.rectangle(image, (x,y), (x+w, y+h), scolor, thickness)

        return image

    def outline_hand(self, image, mask, side, scolor=(255,255,255)):
        image = image.copy()
        mask = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

        x,y = 0,0
        w,h = self.params['dimensions']

        if side in self.params['position']:
            x,y = self.params['position'][side]

            img_crop = image[y:y+h, x:x+w]
            gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
            image[y:y+h, x:x+w] = cv2.add(img_crop, gradient)
        else:
            raise ValueError('only "right" and "left" side available')

        return image