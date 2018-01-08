import torch.utils.data as data

from PIL import Image
import os
import os.path
import cPickle as pickle
import gzip
from scipy.ndimage.interpolation import rotate
from skimage.io import imread
from skimage.transform import resize
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

DAT_EXTENSIONS = [
    '.pickle.gz', '.csv', '.CSV'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_data_file(filename):
    return any(filename.endswith(extension) for extension in DAT_EXTENSIONS)

def make_dataset(dir, fn_filter=None, load_attributes=False):
    '''
    Assumes that the structure of the data folder is:
    dir
        sample_name1.tif (or.jpg, .png etc)   # image data
        sample_name1.pickle.gz (or .csv etc)  # image attributes
    '''
    imgdata = []
    imgattr = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_image_file(fname):
                imgdata.append(path)
            elif is_data_file(fname):
                imgattr.append(path)
            else:
                continue
    # There should be two files for each sample_name : (image, data)
    assert (len(imgdata) == len(imgattr)) or (len(imgattr) == 0)

    # sample names for images should be identical to those for attributes
    # modify this to work with general file extensions
    sample_names_img =[os.path.basename(f).split(".")[0] for f in imgdata]
    sample_names_dat =[os.path.basename(f).split(".")[0] for f in imgattr]
    assert len([f1 for f1,f2 in zip(sample_names_img, sample_names_dat) 
                if f1 != f2]) == 0

    if fn_filter is not None:
        idx = fn_filter(imgdata)
        imgdata = [f for f,i in zip(imgdata,idx) if i]
        imgattr = [f for f,i in zip(imgattr,idx) if i]
        sample_names_img = [f for f,i in zip(sample_names_img,idx) if i]

    if len(imgattr) == 0:
        imgattr = [None for _ in imgdata]
    return zip(imgdata, imgattr, sample_names_img)


def default_loader(path, mode='RGB'):
    return Image.open(path).convert(mode)

def grayscale_loader(path):
    pimg = default_loader(path, mode="L")
    return pimg

def ndimage_loader(path):
    img = imread(path)
    return img

def rotate_pimage(img, max_angle=30):
    theta = (-0.5 + np.random.rand())*max_angle
    return img.rotate(theta, expand=False)

def normalize_ndimage(img0):
    img = img0.copy().astype(float)
    for c in range(img.shape[2]):
        img[...,c] = (img[...,c] - img[...,c].min()) / float((img[...,c].max() - img[...,c].min()))
    return img

def rotate_ndimage(img, max_angle=30):
    '''
    This uses scipy.ndimage and works on images with other than 1 or 3 channels
    '''
    angle = (-0.5 + np.random.rand())*max_angle
    img_rot = rotate(img, angle, reshape=False)
    img_norm = normalize_ndimage(img_rot)
    s = img.max((0,1)) - img.min((0,1))
    img_rot = img.min((0,1)) + img_norm * s
    return img_rot

def flip_ndimage(img, axis=2):
    '''
    This uses scipy.ndimage and works on images with other than 1 or 3 channels
    '''
    if np.random.rand()>0.5:
        return np.flip(img, axis)
    return img

def adjust_range_ndimage(img, n=255, renormalize=False):
    '''
    Converts values in full-range image to 0...255 buckets.
    '''
    img_norm = normalize_ndimage(img)
    img_norm =(img_norm * n).astype(np.uint8)
    return normalize_ndimage(img_norm) if renormalize else img_norm

def attributes_loader(path, fields=None):
    with gzip.open(path, "r") as f:
        dat = pickle.load(f)
    if fields is None:
        return dat
    else:
        return {f:dat[f] for f in fields if f in dat}

def basic_preprocess(img, res, log=False, normalize=False):
    if img.shape[0] != res:
        img = resize(img, (res,res), preserve_range=True)
    img = img.astype(float)

    # only apply transformations to those channels that have non-zero means
    channel_means = img.mean((0,1))
    c = channel_means!=0

    # apply log scale to the requested channels
    if type(log) is not bool:
        c_log = [i for i in log if c[i]]
    elif log:
        c_log = c
    else:
        c_log = None
    if c_log is not None:
        img[...,c_log] = np.log10(img[...,c_log] + 1e-4)
    if type(normalize) is bool:
        img_min = img[...,c].min((0,1))
        img_max = img[...,c].max((0,1))
        scaling = (img_max - img_min).astype(float)
        scaling[scaling==0] = 1.0
        if normalize:
            img[...,c]=(img[...,c] - img_min).astype(float)/scaling
    else: # if normalize is not a flag, assume it's a tuple (mu, sd)
        mu, sd = normalize
        mu = [m for i,m in enumerate(mu) if c[i]]
        sd = [m for i,m in enumerate(sd) if c[i]]
        if log:
            mu = np.array([x if x==0 else np.log10(x) for x in mu])
            sd = np.array([x if x==1 else np.log10(x) for x in sd])
        img[...,c] = (img[...,c] - mu) / sd
    return img

class DataFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 fn_filter=None,
                 image_loader=ndimage_loader, target_loader=None):
        imgs = make_dataset(root, fn_filter=fn_filter)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        print("Found {} images in subfolders of: {}".format(len(imgs), root))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.image_loader = image_loader
        self.target_loader = target_loader

    def __getitem__(self, index):
        path_image, path_attrb, sample_names = self.imgs[index]
        img_data = self.image_loader(path_image)
        img_attr = None if self.target_loader is None or path_attrb is None else self.target_loader(path_attrb)
        if self.transform is not None:
            img_data = self.transform(img_data)
        if self.target_transform is not None and img_attr is not None:
            img_attr = self.target_transform(img_attr)

        if img_attr is None:
            return img_data, sample_names
        else:
            return img_data, img_attr, sample_names

    def __len__(self):
        return len(self.imgs)
