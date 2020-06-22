# import the necessary packages
import numpy as np
import csv
import pickle
import cv2
import math
import operator

class SIFTSearcher:

    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def search(self,queryFeatures, limit = 10):
        kp_q = queryFeatures[0]
        desc_q = queryFeatures[1]
        with open(self.indexPath, 'rb') as f:
            sift_index = pickle.load(f)
        for i in sift_index:
            kp = []
            for point in i[1]:
                temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
                kp.append(temp)
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            desc = i[2]
            matches = flann.knnMatch(desc_q,desc , k=2)
            good_points = []
            ratio = 0.75
            for m, n in matches:
                if m.distance < ratio*n.distance:
                    good_points.append(m)
            i.append(kp)
            i.append(len(good_points))

        sift_index.sort(key = lambda x:x[-1], reverse = True)
        return sift_index[:limit]

class LBPSearcher:

    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def euclidean_distance(self, a,b):
        distance = np.sqrt(np.dot(a - b, a - b))
        return distance

    def search(self, queryFeatures, limit = 10):
        # initialize our dictionary of results
        results = []

        # open the index file for reading
        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # parse out the image ID and features, then compute the
                # chi-squared distance between the features in our index
                # and our query features
                features = [float(x) for x in row[1:]]
                #print( "query:",queryFeatures)
                d = self.chi2_distance(features, queryFeatures)#d = self.cosine_similarity(features, queryFeatures)#d = self.euclidean_distance(features, queryFeatures)#self.chi2_distance(features, queryFeatures)
                #d = self.cosine_similarity(features, queryFeatures)
                # now that we have the distance between the two feature
                # vectors, we can udpate the results dictionary -- the
                # key is the current image ID in the index and the
                # value is the distance we just computed, representing
                # how 'similar' the image in the index is to our query
                results.append((row[0],d))
                #results.append((row[0],d))

            # close the reader
            f.close()

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results.sort(key = operator.itemgetter(1))#results = sorted([(v, k) for (k, v) in results.items()])
        #results.sort(key=operator.itemgetter(1))

        # return our (limited) results
        return results[:limit]



    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d
    def cosine_similarity(self,a, b):
        return sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))
