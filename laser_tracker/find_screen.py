# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import argparse
import cv2

vid = cv2.VideoCapture(1)

def find_screen(vid):
    oldRect = np.zeros((4, 2), dtype = "float32")
    rect = np.zeros((4, 2), dtype = "float32")
    while(True):    
        _, image = vid.read()
        #image = cv2.imread(args["query"])
        ratio = image.shape[0] / 300.0
        orig = image.copy()
        image = imutils.resize(image, height = 300)

        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        thresh = edged.copy()

        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour

        (_,cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None

        # loop over our contours
        for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # if our approximated contour has four points, then
                # we can assume that we have found our screen
                if len(approx) == 4:
                        screenCnt = approx
                        break

        # now that we have our screen contour, we need to determine
        # the top-left, top-right, bottom-right, and bottom-left
        # points so that we can later warp the image -- we'll start
        # by reshaping our contour to be our finals and initializing
        # our output rectangle in top-left, top-right, bottom-right,
        # and bottom-left order
        
        if (screenCnt != None):
            pts = screenCnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype = "float32")

            # the top-left point has the smallest sum whereas the
            # bottom-right has the largest sum
            s = pts.sum(axis = 1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            # compute the difference between the points -- the top-right
            # will have the minumum difference and the bottom-left will
            # have the maximum difference
            diff = np.diff(pts, axis = 1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # multiply the rectangle by the original ratio
            rect *= ratio
        else:
            rect = oldRect
            
        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        cv2.imshow("image", image)
        cv2.imshow("edge", edged)
        cv2.imshow("warp", imutils.resize(warp, height = 300))
        
        # Loop the function until the 'a' key is pressed. This gives time for camera setup. 
        # Return the warp matrix when complete
        if cv2.waitKey(33) == ord('a'):
            cv2.destroyAllWindows()
            return M


def show_warp(M):
    _ , image = vid.read()

    warp = cv2.warpPerspective(image, M, (300, 300))
    cv2.imshow("warp", warp)



warpMatrix = find_screen(vid)
while (True):
    show_warp(warpMatrix)

    if cv2.waitKey(33) == ord('a'):
        cv2.destroyAllWindows()
        break

vid.release()
