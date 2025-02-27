import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa 
import numpy as np
from PIL import Image

from model import CNNModel
from feature_extraction import pre_progressing

from numpy import random
from torchvision import datasets
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader , random_split
from torchvision import models

path ='model_parameter_cnn_{num_epoch}.pth'
model = CNNModel()

#load model 
state_dict = torch.load(path)
model.load_state_dict(state_dict)
model.eval()



def model_test(model, path):
    data, sr = librosa.load(path, sr = 16000)

    mfcc_data = librosa.feature.mfcc(y = data, sr = 16000, n_mfft=40,n_ftt = 400,hop_length=160 )
    mfcc_data = librosa.util.normalize(mfcc_data)
    padding = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
    padding_mfcc = padding(mfcc_data, 400)
    img = Image.fromarray(padding_mfcc)

    img = img.resize((32,32))
    img = img.convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    print(output)
    print(probabilities)

    return output 

path = 'C:/Users/seoji/Desktop/gist/gist_makerton/FSDKaggle2018.audio_train/FSDKaggle2018.audio_train/f08d2ce5.wav'
model_test(model, path)
