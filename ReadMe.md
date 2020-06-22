# CSE 515: MULTIMEDIA AND WEB DATABASES -Phase 1

The main objective of this phase to convert the image to a different vector space using feature descriptors SIFT ( Scale Invariant Feature Transform) and LBP (Local Binary Patterns)  and measuring the similarity between two images using a distance metric.



# Files
**feature_descriptors.py** - Python file which class implementation of SIFT and LBP

**task1.py**-  Python script for Task1

**index.py** - Python script for Task2

**searcher.py** - Python script with class implementation of similarity and matching

**search.py** - Python script for Task3

**small_db** - folder with small set of images

**results/lbp** - folder where the search results will be stored for lbp descriptor

**results/sift** - folder where the search  results will be stored for sift descriptor

**Report.pdf** - Report

# Requirments
The system successfully runs on an windows and linux machines with Anaconda3. Programs leverage use of the pandas package for Dataframes, numpy for the arrays, to ensure the direct functional use of feature descriptors i.e,SIFT and LBP we used packages directly available from the opencv-python and scikit-image libraries respectively. Therefore, one needs to install the following packages:


 **Run the following commands:**
 
     conda create -n myenv python=3.6
     conda activate myenv
     conda install -c menpo opencv
     conda install -c anaconda scipy
     conda install -c anaconda numpy

# Instructions to run
User can run each of the tasks as simple python scripts, with different options as command line arguments.Below are the sample command line executions for each tasks

 **Task1:** -i indicates the input image path, -f indicates the feature

     LBP- python -i small_db/Hand_0008110.jpg -f "lbp"
     SIFT- python -i small_db/Hand_0008110.jpg -f "sift"

**Task2:** Runs through the folder of images specified by "-d"  and stores the features in a file specified by "-i". Descriptor is specified by "-f"

	  LBP- python index.py -d "small_db" -i "lbp-small_db-index.csv" -f lbp
	  SIFT- python index.py -d "small_db" -i "sift-small_db-index.pickle" -f sift

**Task3:** Searches through the  stored feature file specified by "-i" for query image specified by "-q". Finds the k similar images specified by "-l" in the folder specified by "-db". Descriptor is specified by "-f". Saves the similar images in the folder specified by "-r"

    LBP- python search.py -i "lbp-small_db-index.csv" -q queries/Hand_0008111.jpg -db small_db -f lbp -l 3 -r results/lbp
    SIFT- python search.py -i "sift-small_db-index.pickle" -q queries/Hand_0000046.jpg -db small_db -f sift -l 3 -r results/sift
