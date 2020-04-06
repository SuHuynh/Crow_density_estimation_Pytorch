import time
import torch
from dataloader import Image_Loader
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
from Model import Encoder, Decoder
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


# Define hyper-parameter
learning_rate = 0.0001
num_epochs = 40
batch_size = 16
valid_step = 5 # after every 20 iterations, evaluate model once time
plot_step = 10 # after every 20 iterations, save and plot loss of training and testing 


# Load data for training and testing
train_data = Image_Loader(root_path='./data_train.csv', transforms_data=True, aug=True)
test_data = Image_Loader(root_path='./data_test.csv', transforms_data=True, aug=True)
total_train_data = len(train_data)
total_test_data = len(test_data)
print('Number of training data: ', total_train_data)
print('Number of testing data: ', total_test_data)

# Generate the batch in each iteration for training and testing
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

#define model and port to model to gpu if you have gpu
encoder = Encoder(pretrained=True)
encoder = encoder.to(device)

decoder = Decoder()
decoder = decoder.to(device)

encoder.train()
decoder.train()

#define the loss function
criterion = nn.MSELoss()

#using Adam optimizer for encoder and decoder
optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

#TRAINING
iters = 0
everage_training_loss=[]
everage_testing_loss=[]

print('=======> Start Training:')
for epoch in range(num_epochs):
    training_loss = 0.0
    for index, data in enumerate(train_loader):
        iters = iters + 1
        image, label = data
        image = image.to(device)
        label = label.to(device)
        # print(image.max(), label.max())

        features = encoder(image)
        predict_density = decoder(features)
        # print(type(y_pred), type(label))
        # print(y_pred.size(), label.size())
        loss = criterion(predict_density, label)
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        training_loss = training_loss + float(loss.item())

        # evaluate model after every 20 iteration
        if (iters % valid_step) ==0:
            test_loss = 0.0
            correct_pred = 0
            encoder.eval()
            decoder.eval()
            test_iter = 0
            with torch.no_grad():
                for _, data in enumerate(test_loader):
                    test_iter = test_iter + 1
                    image, label = data
                    image = image.to(device)
                    label = label.to(device)
                    features = encoder(image)
                    predict_density = decoder(features)
                    
                    loss = criterion(predict_density, label)
                    test_loss += float(loss.item())

                    output_sample = predict_density[0].data.cpu().numpy().squeeze()
                    output_sample[output_sample<0]=0
                    output_sample = output_sample*255
                    cv2.imwrite('sample_density_output.jpg', output_sample)

            print('Iteration: {}, Training loss: {:.4f}, Test loss: {:.4f}'.format(iters, training_loss/iters, test_loss / test_iter))
            encoder.train()
            decoder.train()

        if (iters % plot_step) == 0:
            everage_training_loss.append(training_loss/iters)
            everage_testing_loss.append(test_loss / test_iter)

            plt.figure(1)
            plt.plot(everage_training_loss, color = 'r') # plotting training loss
            plt.plot(everage_testing_loss, color = 'b') # plotting evaluation loss

            plt.legend(['training loss', 'testing loss'], loc='upper left')
            plt.savefig('plot_loss.png')
            # training_/loss = 0

    # After every epoch, saved model check point once time
    torch.save(encoder.state_dict(), './saved_models/encoder_epoch_{}.pth'.format(epoch+1))
    torch.save(decoder.state_dict(), './saved_models/decoder_epoch_{}.pth'.format(epoch+1))

print('Finished training!')