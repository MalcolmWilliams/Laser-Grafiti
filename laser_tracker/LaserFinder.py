import cv2
import numpy as np
import math


class LaserFinder( object ):

    def __init__(self, img):

        self.drawing = False # true if mouse is pressed
        self.ix,self.iy = -1,-1
        #size = 512
        #img = np.zeros((size,size,3), np.uint8)
        #img = cv2.imread("2016-09-15-171826.jpg", cv2.IMREAD_COLOR)
        self.img_height, self.img_width = img.shape[:2]
        self.target_x, self.target_y = 0, 0
        self.zoom_level = 0
        self.dist = 0
        self.img = img
   
    # mouse callback function
    def draw_circle(self, event,x,y,flags,parami):

        if event == cv2.EVENT_LBUTTONDOWN:
            #drawing = True
            self.ix,self.iy = x,y
            self.target_x += (x-self.img_width/2)/(2**self.zoom_level)
            self.target_y += (y-self.img_height/2)/(2**self.zoom_level)
            
        elif event == cv2.EVENT_LBUTTONUP:
            #drawing = False
            cur_x, cur_y = x, y
            self.dist = math.sqrt( (cur_x-self.ix) **2 + (cur_y-self.iy)**2)
            cv2.circle(self.img,(self.ix,self.iy),int(self.dist),(0,0,255),1)
             
        elif event == cv2.EVENT_RBUTTONDOWN:
            #print "x, y", x,y
            #zoom in on the target x and y coords.
            #img = cv2.resize(img,(2*width, 2*height))
            res = self.img.copy()
            x_min = x-self.img_width/4
            x_max = x+self.img_width/4
            y_min = y-self.img_height/4
            y_max = y+self.img_height/4
            if(x_min < 0): 
                x_max -= x_min
                x_min = 0
            if(y_min < 0):
                y_max -= y_min
                y_min = 0
            if(x_max > self.img_width):
                x_min = x_min - x_max + self.img_width
                x_max = self.img_width
            if(y_max > self.img_height):
                y_min = y_min - y_max + self.img_height
                Y_max = self.img_height
            if(self.zoom_level == 0):
                self.target_x += x
                self.target_y += y
            else:
                self.target_x += (x-self.img_width/2)/ (2**self.zoom_level)
                self.target_y += (y-self.img_height/2)/(2**self.zoom_level)
            self.zoom_level+=1
            #print "target_x, target_y", target_x, target_y
            res = res[ y_min:y_max, x_min:x_max]
            res = cv2.resize(res,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
            #res = res[ (x-size/2):(y-size/2), (x+size/2):(y+size/2) ]
            self.img = res
            #cv2.imshow("res", res)

    def find_laser(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_circle)

        while(1):
            cv2.imshow('image',self.img)
            if cv2.waitKey(33) == ord('a'):
                cv2.destroyAllWindows()
                return self.target_x, self.target_y, (self.dist/2**self.zoom_level)


def cost(img1, img2):
    mult = 10
    img_xor = cv2.bitwise_xor(img1, img2)
    img_and = cv2.bitwise_and(img1, img2)
    #cv2.imshow("img_xor", img_xor)
    #cv2.imshow("img_and", img_and)
    sum_xor = np.sum(img_xor)
    sum_and = np.sum(img_and)
    return sum_and * mult - sum_xor

def get_cost(self, laser_location):
    target_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    


if(__name__ == "__main__"):
    img = cv2.imread("2016-09-15-171826.jpg", cv2.IMREAD_COLOR)
    img_copy = img.copy()
    lf = LaserFinder(img_copy)
    laser = lf.find_laser()
    #print lf

    blank_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    cv2.circle(blank_img,(laser[0], laser[1]),int(laser[2]),(255),-1)
    cv2.circle(img,(laser[0], laser[1]),int(laser[2]),(255),-1)
    cv2.imwrite("image.jpg", blank_img) 
    while(1):
        cv2.imshow('image',img)
        cv2.imshow('blank image',blank_img)
        if cv2.waitKey(33) == ord('a'):
            cv2.destroyAllWindows()
            break
