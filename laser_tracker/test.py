import cv2
import numpy as np
import math

drawing = False # true if mouse is pressed
ix,iy = -1,-1
size = 512

# mouse callback function
def draw_circle(event,x,y,flags,parami):
    global ix,iy,drawing,mode, size

    if event == cv2.EVENT_LBUTTONDOWN:
        #drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        #drawing = False
        cur_x, cur_y = x, y
        dist = math.sqrt( (cur_x-ix) **2 + (cur_y-iy)**2)
        cv2.circle(img,(ix,iy),int(dist),(0,0,255),1)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        print x,y
        #zoom in on the target x and y coords.
        #img = cv2.resize(img,(2*width, 2*height))
        res = img.copy()
        x_min = x-size/4
        x_max = x+size/4
        y_min = y-size/4
        y_max = y+size/4
        if(x_min < 0): 
            x_max -= x_min
            x_min = 0
        if(y_min < 0):
            y_max -= y_min
            y_min = 0
        if(x_max > size):
            x_min = x_min - x_max + size
            x_max = size
        if(y_max > size):
            y_min = y_min - y_max + size
            Y_max = size
        res = res[ y_min:y_max, x_min:x_max]
        res = cv2.resize(res,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        #res = res[ (x-size/2):(y-size/2), (x+size/2):(y+size/2) ]
        cv2.imshow("res", res)

img = np.zeros((size,size,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

'''import cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
print events

import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''
