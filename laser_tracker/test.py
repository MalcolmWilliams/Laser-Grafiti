import cv2
import numpy as np
import math

drawing = False # true if mouse is pressed
ix,iy = -1,-1
#size = 512
#img = np.zeros((size,size,3), np.uint8)
img = cv2.imread("2016-09-15-171826.jpg", cv2.IMREAD_COLOR)
img_height, img_width = img.shape[:2]
target_x, target_y = 0, 0
zoom_level = 0
# mouse callback function
def draw_circle(event,x,y,flags,parami):
    global ix,iy,drawing,mode, img, target_x, target_y, zoom_level, img_height, img_width

    if event == cv2.EVENT_LBUTTONDOWN:
        #drawing = True
        ix,iy = x,y
        target_x += (x-img_width/2)/(2**zoom_level)
        target_y += (y-img_height/2)/(2**zoom_level)
        print "final_x, final_y", target_x, target_y
    elif event == cv2.EVENT_LBUTTONUP:
        #drawing = False
        cur_x, cur_y = x, y
        dist = math.sqrt( (cur_x-ix) **2 + (cur_y-iy)**2)
        cv2.circle(img,(ix,iy),int(dist),(0,0,255),1)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        print "x, y", x,y
        #zoom in on the target x and y coords.
        #img = cv2.resize(img,(2*width, 2*height))
        res = img.copy()
        x_min = x-img_width/4
        x_max = x+img_width/4
        y_min = y-img_height/4
        y_max = y+img_height/4
        if(x_min < 0): 
            x_max -= x_min
            x_min = 0
        if(y_min < 0):
            y_max -= y_min
            y_min = 0
        if(x_max > img_width):
            x_min = x_min - x_max + img_width
            x_max = size
        if(y_max > img_height):
            y_min = y_min - y_max + img_height
            Y_max = size
        if(zoom_level == 0):
            target_x += x
            target_y += y
        else:
            target_x += (x-img_width/2)/ (2**zoom_level)
            target_y += (y-img_height/2)/(2**zoom_level)
        zoom_level+=1
        print "target_x, target_y", target_x, target_y
        res = res[ y_min:y_max, x_min:x_max]
        res = cv2.resize(res,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        #res = res[ (x-size/2):(y-size/2), (x+size/2):(y+size/2) ]
        img = res
        #cv2.imshow("res", res)

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
