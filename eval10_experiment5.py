#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[1]:


get_ipython().system('pip install -q -r requirements.txt')


# In[1]:


import sys
import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import transforms

from ignite.engine import Events
from scripts.ignite import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet

from scripts.evaluate import eval_model, eval_model_10

import warnings
warnings.filterwarnings('ignore')


# ## Define dataset and model

# In[2]:


img_dir = '../input/rxrxairgb512'
path_data = '../input/rxrxaicsv'
device = 'cuda'
batch_size = 32
torch.manual_seed(0)
model_name = 'efficientnet-b3'


# In[3]:


jitter = (0.6, 1.4)
class ImagesDS(D.Dataset):
    # taken textbook from https://arxiv.org/pdf/1812.01187.pdf
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(448),
        transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=.1),
        transforms.RandomHorizontalFlip(p=0.5),
        # PCA Noise should go here,
        transforms.ToTensor(),
        transforms.Normalize(mean=(123.68, 116.779, 103.939), std=(58.393, 57.12, 57.375))
    ])
    
    transform_validation = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(123.68, 116.779, 103.939), std=(58.393, 57.12, 57.375))
    ])

    def __init__(self, df, img_dir=img_dir, mode='train', validation=False, site=1):
        self.records = df.to_records(index=False)
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.validation = validation
        
    @staticmethod
    def _load_img_as_tensor(file_name, validation):
        with Image.open(file_name) as img:
            if not validation:
                return ImagesDS.transform_train(img)
            else:
                return ImagesDS.transform_validation(img)

    def _get_img_path(self, index, site=1):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return f'{self.img_dir}/{self.mode}/{experiment}_{plate}_{well}_s{site}.jpeg'
        
    def __getitem__(self, index):
        img1, img2 = [self._load_img_as_tensor(self._get_img_path(index, site), self.validation) for site in [1,2]]
        if self.mode == 'train':
            return img1, img2, int(self.records[index].sirna)
        else:
            return img1, img2, self.records[index].id_code

    def __len__(self):
        return self.len


class TestImagesDS(D.Dataset):
    transform = transforms.Compose([
        transforms.RandomCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=(123.68, 116.779, 103.939), std=(58.393, 57.12, 57.375))
    ])

    def __init__(self, df, img_dir=img_dir, mode='test', validation=False, site=1):
        self.records = df.to_records(index=False)
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.validation = validation
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return TestImagesDS.transform(img)

    def _get_img_path(self, index, site=1):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return f'{self.img_dir}/{self.mode}/{experiment}_{plate}_{well}_s{site}.jpeg'
        
    def get_image_pair(self, index):
        return [self._load_img_as_tensor(self._get_img_path(index, site)) for site in [1,2]]
    
    def __getitem__(self, index):
        image_pairs = [self.get_image_pair(index) for _ in range(20)]
        
        return image_pairs, self.records[index].id_code

    def __len__(self):
        return self.len


# In[4]:


# dataframes for training, cross-validation, and testing
df_test = pd.read_csv(path_data+'/test.csv')

# pytorch test dataset & loader
ds_test = TestImagesDS(df_test, mode='test', validation=True)
tloader = D.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)


# In[5]:


class EfficientNetTwoInputs(nn.Module):
    def __init__(self):
        super(EfficientNetTwoInputs, self).__init__()
        self.classes = 1108
        
        model = model = EfficientNet.from_pretrained(model_name, num_classes=1108) 
        num_ftrs = model._fc.in_features
        model._fc = nn.Identity()
        
        self.resnet = model
        self.fc = nn.Linear(num_ftrs * 2, self.classes)

    def forward(self, x1, x2):
        x1_out = self.resnet(x1)
        x2_out = self.resnet(x2)
   
        N, _, _, _ = x1.size()
        x1_out = x1_out.view(N, -1)
        x2_out = x2_out.view(N, -1)
        
        out = torch.cat((x1_out, x2_out), 1)
        out = self.fc(out)

        return out 
    
model = EfficientNetTwoInputs()


# #### Evaluate

# In[6]:


model.cuda()
eval_model_10(model, tloader, 'models/Model_efficientnet-b3_93.pth', path_data)

