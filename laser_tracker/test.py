import cv2
import numpy as np


def cost(img1, img2):
    mult = 10
    img_xor = cv2.bitwise_xor(img1, img2)
    img_and = cv2.bitwise_and(img1, img2)
    #cv2.imshow("img_xor", img_xor)
    #cv2.imshow("img_and", img_and)
    sum_xor = np.sum(img_xor)
    sum_and = np.sum(img_and)
    return sum_and * mult - sum_xor




if(__name__ == "__main__"):
    img1 = cv2.imread("image.jpg")
    img2 = cv2.imread("laser_image.jpg")

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    #cv2.imshow("img3", img3)

    print np.sum(img1)
    print np.sum(img2)
    print cost(img1, img2)

    while(1):
        if(cv2.waitKey(100) == ord('q')):
            break

