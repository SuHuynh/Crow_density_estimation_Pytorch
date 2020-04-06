from imgaug import augmenters as iaa
import math
import cv2
import numpy as np
from PIL import Image
from skimage.transform import rotate
from skimage.util import random_noise
import random
# from data_helper import *

def color_augumentor(image, label):

    image = np.array(image)
    label = np.array(label)

    flip_horizon = random.choice([True, False])
    if flip_horizon==True:
        image = np.fliplr(image)
        label = np.fliplr(label)

    flip_vertical = random.choice([True, False])
    if flip_vertical==True:
        image = np.flipud(image)
        label = np.flipud(label)

    rotation = random.choice([True, False])
    if rotation == True:
        image = rotate(image, angle=15)
        label = rotate(label, angle=15)

    add_noise = random.choice([True, False])
    if add_noise==True:
        image=random_noise(image)

    blur = random.choice([True, False])
    if blur==True:
        image = cv2.GaussianBlur(image, (11,11), 0)

    image = Image.fromarray(image)
    return image

def gamma_correction(image):
    image = np.array(image)
    gamma = random.randrange(8, 14)/10
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    image = cv2.LUT(image, lookUpTable)
    image = Image.fromarray(image)
    return image