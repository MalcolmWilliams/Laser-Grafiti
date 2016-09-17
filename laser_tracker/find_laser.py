# Testing ground for various filters to find the laser.

# Desired functionality: 
# 1) Press space to pause webcam feed
# 2) Zoom in and click on laser. (Draw a circle around it?)
# 3) program tunes threshold values (H, S, V) to get best tracking
# 4) Press space to resume normal operation. 


#! /usr/bin/env python
import sys
import argparse
import cv2
import numpy
import perspective_shift



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
        self.createTrackbars()

    def nothing(self,x):
        pass
    
    def createTrackbars(self):
        # Create a black image, a window
        trackbars = numpy.zeros((1,512,3), numpy.uint8)
        cv2.namedWindow('trackbars')

        # create trackbars for hsv ranges
        cv2.createTrackbar('hue_min','trackbars',0,255,self.nothing)
        cv2.createTrackbar('hue_max','trackbars',0,255,self.nothing)
        cv2.createTrackbar('sat_min','trackbars',0,255,self.nothing)
        cv2.createTrackbar('sat_max','trackbars',0,255,self.nothing)
        cv2.createTrackbar('val_min','trackbars',0,255,self.nothing)
        cv2.createTrackbar('val_max','trackbars',0,255,self.nothing)
        
        # create switch for ON/OFF functionality
        #switch = '0 : OFF \n1 : ON'
        #cv2.createTrackbar(switch, 'image',0,1,nothing)

        #while(1):
        cv2.imshow('trackbars',trackbars)
        #k = cv2.waitKey(1) & 0xFF
        #if k == 27:
        #    break


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

    def track(self, frame, mask):
        """
        Track the position of the laser pointer.

        Code taken from
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
        """
        center = None

        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)


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

        # Merge the HSV components back together.
        hsv_image = cv2.merge([
            self.channels['hue'],
            self.channels['saturation'],
            self.channels['value'],
        ])

        #self.track(frame, self.channels['laser'])

        cv2.imshow('Laser', self.channels['laser'])
        cv2.imshow('Hue', self.channels['hue'])
        cv2.imshow('Saturation', self.channels['saturation'])
        cv2.imshow('Value', self.channels['value'])
        cv2.imshow('frame', frame)


        return hsv_image


lt = LaserTracker()
#frame = cv2.imread("2016-09-15-171826.jpg" ,cv2.IMREAD_COLOR)
frame = cv2.imread("Screenshot from 2016-09-15 18-08-29.png" ,cv2.IMREAD_COLOR)

#hsv_image = lt.detect(frame)
#cv2.waitKey(0)

while(1):
    hsv_image = lt.detect(frame)
    #cv2.imshow('trackbars',trackbars)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
