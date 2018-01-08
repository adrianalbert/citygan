import os
import argparse
import glob
import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from datafolder import basic_preprocess, normalize_ndimage
import warnings
warnings.filterwarnings("ignore")

def str2list(v):
    return [int(i) for i in v.split(",")]

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data", help="path to data folder to extract subset from")
parser.add_argument("--extract_path", type=str, default="./", help="path to data folder to create subset in")
parser.add_argument("--take_log", type=str2list, default=None, help="what channels to take logs of")

def extract_subset(args):
    # format paths
    if not os.path.exists(args.extract_path):
        os.makedirs(args.extract_path)
    files = glob.glob(args.data_path + "/*.tif")
    f = files[0]
    C = imread(f).shape[2]
    if not os.path.exists(args.extract_path + "/012/img/"):
        os.makedirs(args.extract_path + "/012/img/")
    for c in range(C):
        if not os.path.exists(args.extract_path + "/" + str(c) + "/img/"):
            os.makedirs(args.extract_path + "/" + str(c) + "/img/")
    # extract files
    print "Processing", len(files), "files", 
    for i,f in enumerate(files):
        if i % 1000 == 0:
            print i,
        img = imread(f)
        W,H,C = img.shape
        img = basic_preprocess(img, W, log=args.take_log, normalize=True)
        fname = os.path.splitext(os.path.basename(f))[0]
        # save first 3 channels
        imsave("%s/012/img/%s.png"%(args.extract_path, fname),img[...,:3])
        # save each individual channel
        for c in range(C):
            imsave("%s/%d/img/%s.png"%(args.extract_path,c,fname),img[...,c])

if __name__ == '__main__':
    # python -W ignore extract_subset.py --data_path=/home/data/world-cities/spatial-maps/samples/ --take_log="1,2" --extract_path=/home/data/world-cities/spatial-maps/splits/
    args, unparsed = parser.parse_known_args()
    extract_subset(args)
