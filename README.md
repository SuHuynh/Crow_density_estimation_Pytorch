# Crow_density_estimation_Pytorch
This repo is for estimate crowd density map and crowd counting with input crowd images

## Data preparation
- Using the ShanghaiTech dataset.
- Link [download](https://www.kaggle.com/tthien/shanghaitech) dataset
- Then, you must use file ```./data_preparation/create_training_set_shtech.m``` to create the density map label
- OR <br>
You can [download](https://drive.google.com/open?id=18jMLVGZrAwuq9p2PY0wlgl0MN_mQXoin) my processed data.
-Then, just copy the data into folder ```./dataset``` like:
<pre>
Crowd_density_estimation
└── dataset/Shanghai_partB
    ├── train_images/
    ├── train_den_images/
</pre>
- run ```python create_file_csv.py``` to create 2 files ```data_train.csv``` and ```data_test.csv```. These files contain all paths to training images and testing images.
## Model
- We use the Autoencoder architecture to train this task. The encoder is adopted from Resnet18 (we remove all fully connected layers). The decoder are built using deconvolutional layers. The encoder and decoder are defined in ```model.py```
## Training
- To train the model, run ```python train.py```
- The checkpoints will saved in folder ```./saved_models``` after every epoch
## Testing
- Modify the path of testing image you want in ```test.py```. Then, run ```python test.py``` and see the results
