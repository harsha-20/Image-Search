from feature_descriptors import LocalBinaryPatterns

from feature_descriptors import Sift
from searcher import LBPSearcher,SIFTSearcher
import argparse
import cv2
import os
import pickle
import shutil
import numpy as np
# construct the argument parser and parse the arguments

def lbp_search():
    desc = LocalBinaryPatterns(8, 2)

    # load the query image and describe it
    query = cv2.imread(args["query"])
    gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    r,c = gray.shape
    hist = []
    lbp = np.zeros((1200,1600))
    for i in range(0,r,100):
        for j in range(0,c,100):
            lbp_buffer,hist_buffer = desc.describe(gray[i:i+100,j:j+100])
            hist.extend(hist_buffer.tolist())
            lbp[i:i+100, j:j+100] = lbp_buffer

    lbp,features = lbp,hist

    # perform the search
    searcher = LBPSearcher(args["index"])
    results = searcher.search(features,limit = int(args["limit"]))
    #print(results)

    # loop over the results
    for i in results:

        # load the result image and display it
        result = cv2.imread(args["imagedb"] + "/" + i[0]+'.jpg')
        resultID = i[0]
        score = i[1]
        #print((args["imagedb"] + "/" + resultID+'.jpg'))
        #print(args["result_path"])
        cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Result', 600,600)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        print("Image:",resultID,'.jpg', "Similarity Measure:",score)

        #print(args["result_path"]+"/"+ resultID+'.jpg',result)
        cv2.imwrite(args["result_path"]+"/"+ resultID+'.jpg',result)

def sift_search():

    sift = Sift()
    query = cv2.imread(args["query"])
    kp_q , desc_q = sift.describe(query)
    searcher = SIFTSearcher(args["index"])
    features = [kp_q,desc_q]
    results = searcher.search(features,limit = int(args["limit"]))

    for i in results:
        result = cv2.imread(args["imagedb"] + "/" + i[0]+'.jpg')
        print("Image:",i[0]+'.jpg', "Similarity Measure(No of key points matched):",i[-1])
        cv2.imwrite(args["result_path"]+"/"+ i[0]+'.jpg',result)
        cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Result', 600,600)
        cv2.imshow("Result", result)
        cv2.waitKey(0)





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--index", required = True,
    	help = "Path to where the computed index will be stored")
    ap.add_argument("-q", "--query", required = True,
    	help = "Path to the query image")
    ap.add_argument("-db", "--imagedb", required = True,
    	help = "Path to the Image Database")
    ap.add_argument("-r", "--result_path", required = True,
    	help = "Path to the store results results")
    ap.add_argument("-f", "--feature", required = True,
        help = "Feature descriptor to be used")
    ap.add_argument("-l", "--limit", required = True,
    	help = "limit of images")
    args = vars(ap.parse_args())
    if not os.path.exists(args["result_path"]):
        os.makedirs(args["result_path"])
    else:
        shutil.rmtree(args["result_path"])
        os.makedirs(args["result_path"])
    if args["feature"] == "lbp":
        lbp_search()
    elif args["feature"] == "sift":
        sift_search()
    else:
        print("Please enter feature argument as 'sift'for SIFT descriptor or 'lbp' for Local Binary Pattern descriptor ")
