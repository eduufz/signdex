# Python
# Downloaded
import numpy as np
import cv2
# SignDex


class Window():
    @classmethod
    def show_tag_panel(cls, tag, count, started):
        # Font config
        img = np.zeros((300,500,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,255,255)
        size = 1
        thickness1 = 1
        thickness2 = 2
        
        """
        text_size = cv2.getTextSize(tag, font, size, thickness)[0]
        text_x = int((img.shape[1] - text_size[0]) / 2)
        text_y = int((img.shape[0] + text_size[1]) / 2)
        """
        
        # Image
        x,y = 15,40
        cv2.putText(img, 'Tag:', (x,y), font, size, color, thickness2, cv2.LINE_AA)
        cv2.putText(img, tag, (x,y*2), font, size, color, thickness1, cv2.LINE_AA)
        cv2.putText(img, 'Total images:', (x,y*3), font, size, color, thickness2, cv2.LINE_AA)
        cv2.putText(img, '{}/{}'.format(count[0],count[1]), (x,y*4), font, size, color, thickness1, cv2.LINE_AA)
        
        if count[0]==count[1]:
            cv2.putText(img, 'Done! Press any button to continue...', (x,y*6), font, 0.5, (0,255,0), thickness1, cv2.LINE_AA)
        elif not started:
            cv2.putText(img, 'Press C to start recording...', (x,y*6), font, 0.5, color, thickness1, cv2.LINE_AA)
        else:
            cv2.putText(img, 'Recording...', (x,y*6), font, 0.5, (0,0,255), thickness1, cv2.LINE_AA)


        # Show image
        cls.show('tag', img)
    
    @staticmethod
    def show(title, image):
        cv2.imshow(title, image)

    @staticmethod
    def clean():
        cv2.destroyAllWindows()