# import the necessary packages
from feature_descriptors import LocalBinaryPatterns
from feature_descriptors import Sift
from imutils import paths
import argparse
import cv2
import os
import numpy as np
import glob
import pickle
import _pickle as cPickle

def sift_index():
    sift_index = []
    for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
        imageID = imagePath[imagePath.rfind("Hand"):-4]
        image = cv2.imread(imagePath)
        sift = Sift()
        kp , desc = sift.describe(image)
        index  = []
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            index.append(temp)
        sift_index.append([imageID,index,desc])

    with open(args["index"],'wb') as  f:
        pickle.dump(sift_index, f)

def lbp_index():
    # initialize the local binary patterns descriptor along with
    # the data and label lists
    desc = LocalBinaryPatterns(8, 2)
    data = []
    labels = []
    hist = []
    lbp = np.zeros((1200,1600))

    # open the output index file for writing
    output = open(args["index"], "w")

    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        imageID = imagePath[imagePath.rfind("Hand"):-4]
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r,c = gray.shape
        hist = []
        for i in range(0,r,100):
            for j in range(0,c,100):
                lbp_buffer,hist_buffer = desc.describe(gray[i:i+100,j:j+100])
                hist.extend(hist_buffer.tolist())
                lbp[i:i+100,j:j+100] = lbp_buffer


        # write the features to file
        features = [str(f) for f in hist]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

    # close the index file
    output.close()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required = True,
    	help = "Path to the directory that contains the images to be indexed")
    ap.add_argument("-i", "--index", required = True,
    	help = "Path to where the computed index will be stored")
    ap.add_argument("-f", "--feature", required = True,
        help = "Feature descriptor to be used")
    args = vars(ap.parse_args())
    if args["feature"] == "lbp":
        lbp_index()
    elif args["feature"] == "sift":
        sift_index()
    else:
        print("Please enter feature argument as 'sift'for SIFT descriptor or 'lbp' for Local Binary Pattern descriptor ")
