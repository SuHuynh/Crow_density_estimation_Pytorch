import csv
import os
from imutils import paths
import random

img_folder = './dataset/Shanghai_partB/train_images/'
label_folder = './dataset/Shanghai_partB/train_den_images/'
image_name_list = os.listdir(label_folder)

with open('data_train.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(320):
        for j in range(15):
            if "{}_{}.png".format(i+1, j+1) in image_name_list:
                image_path = img_folder + "{}_{}.jpg".format(i+1, j+1)
                label_path = label_folder + "{}_{}.png".format(i+1, j+1)
                writer.writerow([image_path, label_path])

with open('data_test.csv', 'a') as csvfile:

    writer = csv.writer(csvfile)
    for i in range(320, 400):
        for j in range(15):
            if "{}_{}.png".format(i+1, j+1) in image_name_list:
                image_path = img_folder + "{}_{}.jpg".format(i+1, j+1)
                label_path = label_folder + "{}_{}.png".format(i+1, j+1)
                writer.writerow([image_path, label_path])
        
