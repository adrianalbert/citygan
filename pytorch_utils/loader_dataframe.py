# Adrian Albert
# 2017
# adapted from 
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

import torch.utils.data as data
import torch

from PIL import Image
from skimage.io import imread
import os
import os.path
import pandas as pd
import numpy as np
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def parse_list(s):
    s = re.sub('\s+', ' ', s[1:-1]).strip().replace(",",' ')
    s = re.sub('\s+', ' ', s)
    # print s.split(" ")
    if len(s.split(" "))>0:
        ret = [float(n.strip()) for n in s.split(" ")]
    return ret


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(df, label_columns=None):
    '''
    Create dictionaries of classes from categorical columns in dataframe.
    '''
    if label_columns is None:
        return None, None
    categCols = [c for c in label_columns \
                    if df[c].dtype==object and type(df[c].iloc[0])==str]
    if len(categCols) == 0:
        classes = None
        class_to_idx = None
    else:
        classes = {}
        class_to_idx = {}
        for c in categCols:
            classes[c] = df[c].unique().tolist()
            classes[c].sort()
            class_to_idx[c] = {classes[c][i]:i for i in range(len(classes[c]))}
    return classes, class_to_idx


def make_dataset(df, filenameCol="filename", label_columns=None, class_to_idx=None):
    if type(filenameCol) == str:
        filenameCol = [filenameCol]
    for c in filenameCol:
        df = df[df[c].apply(is_image_file)]
    categ_labels = class_to_idx.keys() if class_to_idx is not None else []
    filenames = df[filenameCol].values
    if label_columns is not None:
        labels = []
        for c in label_columns:
            cur_labels = df[c].values
            if c in categ_labels:
                cur_labels = map(lambda x: class_to_idx[c][x], cur_labels)
            labels.append(cur_labels)
        images = zip(filenames, zip(*labels))
    else:
        images = zip(filenames, [-1 for _ in filenames])
    return images


# def default_loader(path, mode="RGB"):
#     '''
#         mode can be either "RGB" or "L" (grayscale)
#     '''
#     return Image.open(path).convert(mode)


def default_loader(path, mode="RGB"):
    '''
        mode can be either "RGB" or "L" (grayscale)
    '''
    img = imread(path).astype(np.uint8)
    if mode == 'RGB' and len(img.shape)==2:
        img = np.array([img, img, img]).transpose([1,2,0])
    return Image.fromarray(img, mode=mode)

def remove_nodata(pimg, val_nodata=128):
    from collections import Counter
    img = np.array(pimg)
    img[abs(img-val_nodata)<0.01] = 1 # hack to remove no-data patches
    pimg = Image.fromarray(np.uint8(img))  
    return pimg

def grayscale_loader(path, val_nodata=None):
    pimg = default_loader(path, mode="L")
    if val_nodata is not None:
        return remove_nodata(pimg)
    else:
        return pimg


def fn_rotate(img, max_angle=30):
    theta = (-0.5 + np.random.rand())*max_angle
    return img.rotate(theta, expand=False)


class ImageDataFrame(data.Dataset):
    '''
    Assumes a Pandas dataframe input with the following columns:
        filename, label_columns

    '''
    def __init__(self, df, transform=None, target_transform=None,
                 loader=default_loader, filenameCol="filename", 
                 label_columns=None, return_paths=False, **kwargs):
        if type(df) == str:
            print "loading from file", df
            df = pd.read_csv(df)
            if label_columns is not None:
                for c in label_columns:
                    if df[c].dtype != object:
                        continue
                    try:
                        df[c] = df[c].apply(parse_list)
                    except:
                        pass

        classes, class_to_idx = find_classes(df, label_columns)
        imgs = make_dataset(df, filenameCol=filenameCol, 
            label_columns=label_columns, class_to_idx=class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in dataframe of: "+len(df)+"\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths

    def __getitem__(self, index):
        paths, labels = self.imgs[index]
        loaders = self.loader
        if type(loaders)!=list:
            loaders = [loaders]
        img = [loaders[j](p) for j,p in enumerate(paths)]
        if self.transform is not None:
            img = [self.transform(i) for i in img]
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        if len(img) == 1:
            img = img[0]
            paths = paths[0]
        else:
            paths = paths.tolist()

        if self.return_paths:
            return img, labels, paths
        else:
            return img, labels
   

    def __len__(self):
        return len(self.imgs)
