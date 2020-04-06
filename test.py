import torch
from Model import Encoder, Decoder
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Define hyper-parameter
img_size = (512, 384)

#define model
encoder = Encoder()
decoder = Decoder()
pre_trained_encoder = torch.load('./saved_models/encoder_epoch_20.pth')
pre_trained_decoder = torch.load('./saved_models/decoder_epoch_20.pth')
encoder.load_state_dict(pre_trained_encoder)
decoder.load_state_dict(pre_trained_decoder)

#port to model to gpu if you have gpu
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

# load and pre-process testing image
# Note: you need to precess testing image similarly to the training images 
img_path = './test_img_2.jpg'
img = cv2.imread(img_path)
print(img.shape)

# resize img to 48x48
img = cv2.resize(img, img_size)
print(img.shape)

# normalize img from [0, 255] to [0, 1]
img = img/255
img = img.astype('float32')

# convert image to torch with size (1, 1, 48, 48)
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0)

with torch.no_grad():
    img = img.to(device)
    features = encoder(img)
    density_pre = decoder(features)           
    density_pre = density_pre.data.cpu().numpy().squeeze()

    # calculate number of people in the image
    density_pre[density_pre<0]=0
    number_people = density_pre.sum()*0.08

    # denormalize density output
    density_pre = density_pre*255
    cv2.imwrite('test_density_output.jpg', density_pre)

    # cv2.putText(density_pre, 'number of peole: ' + str(number_people), (20, 20),
    #                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
    # cv2.imshow('density output', np.array(density_pre))
    # cv2.waitKey(0)
    plt.imshow(density_pre)
    plt.title('Testing_Density_Output')
    plt.xlabel('Estimation_number_of_people: {:.2f}'.format(number_people))
    plt.show()
