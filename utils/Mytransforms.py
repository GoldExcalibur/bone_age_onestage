from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import types
import collections
import warnings
import cv2
import copy
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

border_value = 0
def normalize(tensor, mean, std):
    """Normalize a ``torch.tensor``

    Args:
        tensor (torch.tensor): tensor to be normalized.
        mean: (list): the mean of BGR
        std: (list): the std of BGR
    
    Returns:
        Tensor: Normalized tensor.
    """
    
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def instance_normalize(tensor):
    c, h, w = tensor.shape
    tensor_t = tensor.reshape(c, h*w)
    mean = torch.mean(tensor_t, dim = 1)
    std = torch.std(tensor_t, dim = 1)
    return normalize(tensor, mean, std)

def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    img = torch.from_numpy(pic.transpose((2, 0, 1)))

    return img.float()

def resize(img, ori_kpt, ratio):
    """Resize the ``numpy.ndarray`` and points as ratio.

    Args:
        img    (numpy.ndarray):   Image to be resized.
        kpt    (list):            Keypoints to be resized.
        ratio  (tuple or number): the ratio to resize.

    Returns:
        numpy.ndarray: Resized image.
        lists:         Resized keypoints.
    """
    
    if not (isinstance(ratio, numbers.Number) or (isinstance(ratio, collections.Iterable) and len(ratio) == 2)):
        raise TypeError('Got inappropriate ratio arg: {}'.format(ratio))
    
    h, w, _ = img.shape
    kpt = copy.deepcopy(ori_kpt)
    if isinstance(ratio, numbers.Number):

        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio
            kpt[i][1] *= ratio

        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio),  kpt 

    else:
        num = len(kpt)
        for i in range(num):
            kpt[i][0] *= ratio[0]
            kpt[i][1] *= ratio[1]
            
      
        return np.ascontiguousarray(cv2.resize(img,(0, 0), fx=ratio[0], fy=ratio[1])), kpt 

class Resized(object):
    """Resize the given numpy.ndarray to target size 

    Args:
        target_size: the target size to resize.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    @staticmethod
    def get_params(img, target_size):

        height, width, _ = img.shape
        ratio = [float(target_size[0]) / width, float(target_size[1]) / height]

        return ratio

    def __call__(self, img, kpt):
        """
        Args:
            img     (numpy.ndarray): Image to be resized.
            kpt     (list):          keypoints to be resized.

        Returns:
            numpy.ndarray: resize image.
            list:          resize keypoints.
        """
        ratio = self.get_params(img,  self.target_size)

        return resize(img, kpt, ratio)
    
class RandomJitter(object):
    def __init__(self, prob_list):
        self.prob_list = prob_list
        self.factor_list = [1.0, 1.2, 1.4, 1.6]
    def __call__(self, img, kpt):
        factor = float(np.random.choice(self.factor_list, size =1, p = self.prob_list))
        img_img = Image.fromarray(img)
        img_jitter = np.array(F.adjust_brightness(img_img, factor))
        return img_jitter, kpt
    
class RandomDrop(object):
    def __init__(self, prob_list):
        self.ratio_list = [1.0, 0.8, 0.6]
        self.prob_list= prob_list

    def __call__(self, img, kpt):
        ratio = float(np.random.choice(self.ratio_list, size = 1, p = self.prob_list))
        h, w, _ = img.shape
        if ratio == 1.0:
            return img, kpt
        
        img_resize, kpt_resize = resize(img, kpt, ratio)
        new_img = np.full(img.shape, border_value)
        h_resize, w_resize, _ = img_resize.shape
        start_h = np.random.randint(0, h - h_resize, 1)
        start_w = np.random.randint(0, w - w_resize, 1)
        ww, hh = np.meshgrid(range(start_w, start_w + w_resize), range(start_h, start_h + h_resize))
        kpt_new = copy.deepcopy(kpt_resize)
        for i in range(len(kpt)):
            kpt_new[i][0] += start_w
            kpt_new[i][1] += start_h
        new_img[hh, ww] = img_resize
        return new_img.astype(np.uint8), kpt_new
    

class TestResized(object):
    """Resize the given numpy.ndarray to the size for test.

    Args:
        size: the size to resize.
    """

    def __init__(self, size):
        assert (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        height, width, _ = img.shape
        
        return (output_size[0] * 1.0 / width, output_size[1] * 1.0 / height)

    def __call__(self, img, kpt):
        """
        Args:
            img     (numpy.ndarray): Image to be resized.
            kpt     (list):          keypoints to be resized.

        Returns:
            numpy.ndarray: Randomly resize image.
            numpy.ndarray: Randomly resize mask.
            list:          Randomly resize keypoints.
            list:          Randomly resize center points.
        """
        ratio = self.get_params(img, self.size)

        return resize(img, kpt, ratio)

def rotate(img,  ori_kpt, degree):
    """Rotate the ``numpy.ndarray`` and points as degree.

    Args:
        img    (numpy.ndarray): Image to be rotated.
        kpt    (list):          Keypoints to be rotated.
        degree (number):        the degree to rotate.

    Returns:
        numpy.ndarray: Resized image.
        numpy.ndarray: Resized mask.
        list:          Resized keypoints.
        list:          Resized center points.
    """
    kpt = copy.deepcopy(ori_kpt)
    height, width, _ = img.shape
    
    img_center = (width / 2.0 , height / 2.0)
    
    rotateMat = cv2.getRotationMatrix2D(img_center, degree, 1.0)
    cos_val = np.abs(rotateMat[0, 0])
    sin_val = np.abs(rotateMat[0, 1])
    new_width = int(height * sin_val + width * cos_val)
    new_height = int(height * cos_val + width * sin_val)
    rotateMat[0, 2] += (new_width / 2.) - img_center[0]
    rotateMat[1, 2] += (new_height / 2.) - img_center[1]


    img = cv2.warpAffine(img, rotateMat, (new_width, new_height), borderValue=(border_value, border_value, border_value))

    num = len(kpt)
    for i in range(num):
        x = kpt[i][0]
        y = kpt[i][1]
        p = np.array([x, y, 1])
        p = rotateMat.dot(p)
        kpt[i][0] = p[0]
        kpt[i][1] = p[1]

    return np.ascontiguousarray(img), kpt 

class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree, prob):
        assert isinstance(max_degree, numbers.Number)
        self.max_degree = max_degree
        self.prob = prob

    @staticmethod
    def get_params(max_degree):
        """Get parameters for ``rotate`` for a random rotate.

        Returns:
            number: degree to be passed to ``rotate`` for random rotate.
        """
        degree = random.uniform(-max_degree, max_degree)

        return degree

    def __call__(self, img, kpt):
        """
        Args:
            img    (numpy.ndarray): Image to be rotated.
            kpt    (list):          Keypoints to be rotated.

        Returns:
            numpy.ndarray: Rotated image.
            list:          Rotated key points.
        """
        if random.random() < self.prob:
            return img, kpt
        
        degree = self.get_params(self.max_degree)

        return rotate(img, kpt, degree)

def vflip(img, ori_kpt):

    height, width, _ = img.shape
    kpt = copy.deepcopy(ori_kpt)
    img = img[::-1, :, :]
    # the number of keypoint
    num = len(kpt)
    for i in range(num):
        kpt[i][1] = height - 1 - kpt[i][1]

    return np.ascontiguousarray(img), kpt 

class RandomVerticalFlip(object):
    """Random vertical flip the image.

    Args:
        prob (number): the probability to flip.
    """
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, kpt):
        """
        Args:
            img    (numpy.ndarray): Image to be flipped.
            kpt    (list):          Keypoints to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < self.prob:
            return vflip(img, kpt)
        return img, kpt 

def hflip(img, ori_kpt):

    height, width, _ = img.shape
    kpt = copy.deepcopy(ori_kpt)
    img = img[:, ::-1, :]

    num = len(kpt)
    for i in range(num):
        kpt[i][0] = width - 1 - kpt[i][0]

    return np.ascontiguousarray(img), kpt 

class RandomHorizontalFlip(object):
    """Random horizontal flip the image.

    Args:
        prob (number): the probability to flip.
    """
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img, kpt):
        """
        Args:
            img    (numpy.ndarray): Image to be flipped.
            kpt    (list):          Keypoints to be flipped.

        Returns:
            numpy.ndarray: Randomly flipped image.
            list: Randomly flipped points.
        """
        if random.random() < self.prob:
            return hflip(img, kpt)
        return img, kpt 

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Mytransforms.Compose([
        >>>     Mytransforms.CenterCrop(10),
        >>>     Mytransforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpt):

        for t in self.transforms:
            img, kpt = t(img, kpt)

        return img, kpt 
