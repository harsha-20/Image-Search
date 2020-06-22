from feature_descriptors import LocalBinaryPatterns
from feature_descriptors import Sift
import argparse
import cv2
import os
import numpy as np

def lbp_show():
    desc = LocalBinaryPatterns(8, 2)
    imagePath = args["imagePath"]
    image = cv2.imread(imagePath)
    lbp = np.zeros((1200,1600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    r,c = gray.shape
    hist = []
    for i in range(0,r,100):
        for j in range(0,c,100):
            lbp_buffer,hist_buffer = desc.describe(gray[i:i+100,j:j+100])
            hist.extend(hist_buffer.tolist())
            lbp[i:i+100,j:j+100] = lbp_buffer
    print("LBP codes:", hist)
    cv2.namedWindow('LBP Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('LBP Image', 600,600)
    cv2.imshow('LBP Image', lbp)
    cv2.waitKey(0)
def sift_show():
    sift = Sift()
    image = cv2.imread(args["imagePath"])
    kp , desc = sift.describe(image)
    img = cv2.drawKeypoints(image, kp, None)

    cv2.namedWindow('SIFT Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SIFT Image', 600,600)
    cv2.imshow("SIFT Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagePath", required = True,
    	help = "Input Image")
    ap.add_argument("-f", "--feature", required = True,
        help = "Feature descriptor to be used")
    args = vars(ap.parse_args())
    if args["feature"] == "lbp":
        lbp_show()
    elif args["feature"] == "sift":
        sift_show()
    else:
        print("Please enter feature argument as 'sift'for SIFT descriptor or 'lbp' for Local Binary Pattern descriptor ")
