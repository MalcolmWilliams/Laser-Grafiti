# Testing ground for various filters to find the laser.

# Functionality
# 1) A still is supplied.
# 2) Zoom in on the laser (Right click)
# 3) Draw a cricle around the laser (Left Click and Drag)
# 4) Hit 'a' to accept
# 5) It will autotune the HSV thresholds.
# 6) Manual touchups can be done if the autotuning is not succesful.

import sys
import argparse
import cv2
import numpy
import perspective_shift
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




class LaserTracker(object):
    def __init__(self, cam_width=640, cam_height=480, hue_min=20, hue_max=160,
                 sat_min=100, sat_max=255, val_min=200, val_max=256,
                 display_thresholds=False,device_num=0):
        """
        * ``cam_width`` x ``cam_height`` -- This should be the size of the
        image coming from the camera. Default is 640x480.

        HSV color space Threshold values for a RED laser pointer are determined
        by:

        * ``hue_min``, ``hue_max`` -- Min/Max allowed Hue values
        * ``sat_min``, ``sat_max`` -- Min/Max allowed Saturation values
        * ``val_min``, ``val_max`` -- Min/Max allowed pixel values

        If the dot from the laser pointer doesn't fall within these values, it
        will be ignored.

        * ``display_thresholds`` -- if True, additional windows will display
          values for threshold image channels.

        """

        self.cam_width = cam_width
        self.cam_height = cam_height
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.display_thresholds = display_thresholds
        self.device_num = device_num        

        self.capture = None  # camera capture device
        self.channels = {
            'hue': None,
            'saturation': None,
            'value': None,
            'laser': None,
        }

        self.previous_position = None
        self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                 numpy.uint8)

    def set_thresh(self, thresh_values):
        self.hue_min = thresh_values[0]
        self.hue_max = thresh_values[1]
        self.sat_min = thresh_values[2]
        self.sat_max = thresh_values[3]
        self.val_min = thresh_values[4]
        self.val_max = thresh_values[5]

    def get_thresh(self):
        return self.hue_min, self.hue_max, self.sat_min, self.hue_max, self.val_min, self.val_max

    def nothing(self,x):
        pass

    def createTrackbars(self):
        # Create a black image, a window
        trackbars = numpy.zeros((1,512,3), numpy.uint8)
        cv2.namedWindow('trackbars')

        # create trackbars for hsv ranges
        cv2.createTrackbar('hue_min','trackbars',int(self.hue_min),256,self.nothing)
        cv2.createTrackbar('hue_max','trackbars',int(self.hue_max),256,self.nothing)
        cv2.createTrackbar('sat_min','trackbars',int(self.sat_min),256,self.nothing)
        cv2.createTrackbar('sat_max','trackbars',int(self.sat_max),256,self.nothing)
        cv2.createTrackbar('val_min','trackbars',int(self.val_min),256,self.nothing)
        cv2.createTrackbar('val_max','trackbars',int(self.val_max),256,self.nothing)
        cv2.imshow('trackbars',trackbars)

    def threshold_image(self, channel):
        if channel == "hue":
            minimum = self.hue_min
            maximum = self.hue_max
        elif channel == "saturation":
            minimum = self.sat_min
            maximum = self.sat_max
        elif channel == "value":
            minimum = self.val_min
            maximum = self.val_max

        (t, tmp) = cv2.threshold(
            self.channels[channel],  # src
            maximum,  # threshold value
            0,  # we dont care because of the selected type
            cv2.THRESH_TOZERO_INV  # t type
        )

        (t, self.channels[channel]) = cv2.threshold(
            tmp,  # src
            minimum,  # threshold value
            255,  # maxvalue
            cv2.THRESH_BINARY  # type
        )

        if channel == 'hue':
            # only works for filtering red color because the range for the hue
            # is split
            self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])

            
    def auto_detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # Threshold ranges of HSV components; storing the results in place
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation'],
            self.channels['laser']
        )

        return (self.channels['hue']).astype(numpy.uint8),  (self.channels['saturation']).astype(numpy.uint8),  (self.channels['value']).astype(numpy.uint8),  (self.channels['laser']).astype(numpy.uint8)

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v

        # get current positions of four trackbars
        self.hue_min = cv2.getTrackbarPos('hue_min','trackbars')
        self.hue_max = cv2.getTrackbarPos('hue_max','trackbars')
        self.sat_min = cv2.getTrackbarPos('sat_min','trackbars')
        self.sat_max = cv2.getTrackbarPos('sat_max','trackbars')
        self.val_min = cv2.getTrackbarPos('val_min','trackbars')
        self.val_max = cv2.getTrackbarPos('val_max','trackbars')
        
        # Threshold ranges of HSV components; storing the results in place
        self.threshold_image("hue")
        self.threshold_image("saturation")
        self.threshold_image("value")

        # Perform an AND on HSV components to identify the laser!
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue'],
            self.channels['value']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation'],
            self.channels['laser']
        )
        '''
        # Merge the HSV components back together.
        hsv_image = cv2.merge([
            self.channels['hue'],
            self.channels['saturation'],
            self.channels['value'],
        ])'''

        #self.track(frame, self.channels['laser'])

        cv2.imshow('Laser', self.channels['laser'])
        cv2.imshow('Hue', self.channels['hue'])
        cv2.imshow('Saturation', self.channels['saturation'])
        cv2.imshow('Value', self.channels['value'])
        cv2.imshow('frame', frame)
        return self.hue_min, self.hue_max, self.sat_min, self.sat_max, self.val_min, self.val_max 

    def make_laser_mask(self, laser_location):
        self.laser_mask = numpy.zeros((self.cam_height, self.cam_width), numpy.uint8)
        cv2.circle(self.laser_mask,(laser_location[0], laser_location[1]),int(laser_location[2]),(255),-1)
        cv2.imshow("laser_mask", self.laser_mask) 

    def set_frame(self, frame):
        self.frame = frame
        self.cam_width = frame.shape[1]
        self.cam_height = frame.shape[0]

    def cost(self, img1, img2):
        #returning the lowest value will give the best result
        mult = 1000
        '''
        print type(img1)
        print type(img2)
       
        print len(img1)
        print len(img2)

        print img1.shape
        print img2.shape
        '''
        img_xor = cv2.bitwise_xor(img1, img2)
        img_and = cv2.bitwise_and(img1, img2)
        #cv2.imshow("img_xor", img_xor)
        #cv2.imshow("img_and", img_and)
        sum_xor = numpy.sum(img_xor)
        sum_and = numpy.sum(img_and)
        return sum_xor - (sum_and * mult)

    def get_cost_hue(self):
        return self.cost(self.laser_mask, self.auto_detect(self.frame)[0]) 
    
    def get_cost_sat(self):
        return self.cost(self.laser_mask, self.auto_detect(self.frame)[1]) 
    
    def get_cost_val(self):
        return self.cost(self.laser_mask, self.auto_detect(self.frame)[2]) 
    
    def get_cost_laser(self):
        return self.cost(self.laser_mask, self.auto_detect(self.frame)[3]) 

    def tune_hue(self):
        self.hue_min = 0
        self.hue_max = 256
        last_cost = self.get_cost_hue()
        while(1):
            cost = self.get_cost_hue()
            if(last_cost < cost):
                self.hue_max = self.hue_max + 1
                break
            self.hue_max = self.hue_max -1
            last_cost = cost
            if (self.hue_max == self.hue_min):
                return self.hue_min, self.hue_max+1
        return self.hue_min, self.hue_max

    def tune_sat(self):
        self.sat_min = 0
        self.sat_max = 256
        last_cost = self.get_cost_sat()
        self.sat_max= 255
        while(1):
            cost = self.get_cost_sat() 
            if (last_cost < cost):
                self.sat_max = self.sat_max+1
                break
            self.sat_max = self.sat_max - 1
            last_cost = cost
            if(self.sat_max == 1):
                return self.sat_min, self.sat_max
        #last_cost = self.get_cost_sat()
        while(1):
            cost = self.get_cost_sat()
            if (last_cost < cost):
                self.sat_min = self.sat_min-1
                break
            self.sat_min =self.sat_min + 1
            last_cost = cost
            if (self.sat_min == self.sat_max):
                return self.sat_min -1, self.sat_max
        return self.sat_min, self.sat_max

    def tune_val(self):
        self.val_max = 256
        self.val_min = 0
        last_cost = self.get_cost_val()
        while(1):
            cost = self.get_cost_val()
            if(last_cost < cost):
                self.val_min = self.val_min - 1
                break
            self.val_min = self.val_min+1
            last_cost = cost
            if(self.val_min == self.val_max):
                return self.val_min -1, self.val_max
        return self.val_min, self.val_max


def manual_tune(frame, thresh_current):
    lt = LaserTracker()
    lt.createTrackbars()
    lt.set_thresh(thresh_current)
    while(1):
        thresh_vals = lt.detect(frame)
        if cv2.waitKey(33) == ord('a'):
            cv2.destroyAllWindows()
            return thresh_vals

def auto_tune(frame, thresh_current):
    limits = ( (0, 256), (0, 256),(0, 256),(0, 256),(0, 256),(0, 256))
    limits2 = ( (0, 256), (0, 256))

    lt = LaserTracker()
    lt.set_frame(frame)
     
    lf = LaserFinder(frame) 
    laser_location = lf.find_laser();
    #print laser_location
    #laser_location = (197, 249, 3)   #For testing, uncomment this line and comment the one above
    lt.make_laser_mask(laser_location)
   
    
    #print "pre optimization:", thresh_current

    thresh_hue = lt.tune_hue()
    thresh_sat = lt.tune_sat()
    thresh_val = lt.tune_val()

    thresh_vals = numpy.concatenate((thresh_hue, thresh_sat, thresh_val))
    lt.set_thresh(thresh_vals)
    
    #print "post optimization:", thresh_vals
    #print res.thresh_current
    lt.createTrackbars()
    while(1):
        thresh_vals = lt.detect(frame)

        if cv2.waitKey(33) == ord('a'):
            cv2.destroyAllWindows()
            return thresh_vals




if ( __name__ == "__main__"):
    frame = cv2.imread("autotune_laser_test_image.jpg")
    #manual_tune(frame, (20, 160, 100, 255, 200, 256) )
    auto_tune(frame, (20, 160, 100, 255, 200, 256) )
