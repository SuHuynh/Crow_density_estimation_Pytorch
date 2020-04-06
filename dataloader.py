import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from augmentation import gamma_correction, color_augumentor
import random

class Image_Loader(Dataset):
    def __init__(self, root_path='./data_train.csv', transforms_data=True, aug=True):
        
        self.data_path = pd.read_csv(root_path)
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        self.aug = aug
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, item):

        # read crowd image 
        image_path = os.path.join(self.data_path.iloc[item, 0])
        image = Image.open(image_path)
        
        # read density label
        label_path = os.path.join(self.data_path.iloc[item, 1])
        density_label = Image.open(label_path)

        # Augment the data
        if self.aug==True:

            gamma_correction_flag = random.choice([True, False])
            color_augumentor_flag = random.choice([True, False])

            if gamma_correction_flag==True:
                image = gamma_correction(image)
            # if color_augumentor_flag==True:
            #     image = color_augumentor(image,density_label)

        if self.transforms_data == True:
            data_transform = self.transform(True)
            image = data_transform(image)
            density_label =data_transform(density_label)

        return image, density_label  #torch.from_numpy(np.array(density_label, dtype=np.float32))

    def transform(self, totensor):
        options = []

        if totensor:
            options.append(transforms.ToTensor())
        
        transform = transforms.Compose(options)

        return transform