# Python
# Downloaded
import numpy as np
import cv2
# SignDex


class Window():
    @staticmethod
    def show_tag_panel(tag, count, started):
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
        cv2.imshow('tag_panel', img)
    
    @staticmethod
    def show_tag_prediction(tag):
        prediction_box  = np.zeros((200,200), np.uint8)
        
        # TEXT FORMAT
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255,255,255)
        size = 3
        thickness = 6
        
        text_size = cv2.getTextSize(tag, font, size, thickness)[0]
        
        if tag == 'SPACE' or tag == 'DELETE':
            text_size = (141,50)
            tag = tag.strip()
            size = 1.5
            thickness = 2
        
        x = int((prediction_box.shape[1] - text_size[0]) / 2)
        y = int((prediction_box.shape[0] + text_size[1]) / 2)

        if tag == 'SPACE': x -= 5
        if tag == 'DELETE': x -= 10
        
        # IMAGE
        cv2.putText(prediction_box, tag, (x,y), font, size, color, thickness, cv2.LINE_AA)
        
        # Show image
        cv2.imshow('tag_prediction', prediction_box)

    @staticmethod
    def clean():
        cv2.destroyAllWindows()