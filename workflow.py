#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[2]:


get_ipython().system('pip install -q -r requirements.txt')


# In[3]:


import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rxrxutils.rxrx.io as rio
from scipy import misc

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

from torchvision import models, transforms

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline


# In[26]:


learning_rate_str, model_name = sys.argv[1:] if len(sys.argv) >= 3 else ['30e-5', 'resnet18']
learning_rate = float(learning_rate_str)

print(f'learning rate: {learning_rate}')
print(f'model name: {model_name}')


# ## Define dataset and model

# In[4]:


path_data = '../input/rxrxai'
device = 'cuda'
batch_size = 32
torch.manual_seed(0)


# In[12]:


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir=path_data, mode='train', site=1, channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return transforms.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
        
    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len


# In[13]:


# dataframes for training, cross-validation, and testing
df = pd.read_csv(path_data+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.05, random_state=42)
df_test = pd.read_csv(path_data+'/test.csv')

# pytorch training dataset & loader
ds = ImagesDS(df_train, mode='train')
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

# pytorch cross-validation dataset & loader
ds_val = ImagesDS(df_val, mode='train')
val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=0)

# pytorch test dataset & loader
ds_test = ImagesDS(df_test, mode='test')
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)


# In[14]:


classes = 1108

model = getattr(models, model_name)(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, classes)

# let's make our model work with 6 channels
trained_kernel = model.conv1.weight
new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
model.conv1 = new_conv


# In[15]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[16]:


metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


# In[17]:


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch, 
                      metrics['loss'], 
                      metrics['accuracy']))


# In[18]:


lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

def adjust_learning_rate(optimizer, epoch):
    # inspired by https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
    lr = learning_rate
    if epoch > 25:
        lr = learning_rate / 2.
    if epoch > 30:
        lr = learning_rate / 4.
    if epoch > 35:
        lr = learning_rate / 10.
    if epoch > 40:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@trainer.on(Events.EPOCH_COMPLETED)
def update_lr_scheduler(engine):
    adjust_learning_rate(optimizer, engine.state.epoch)
    lr = float(optimizer.param_groups[0]['lr'])
    print("Learning rate: {}".format(lr))


# In[19]:


@trainer.on(Events.EPOCH_STARTED)
def turn_on_layers(engine):
    epoch = engine.state.epoch
    if epoch == 1:
        for name, child in model.named_children():
            if name == 'fc':
                pbar.log_message(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                pbar.log_message(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
    if epoch == 3:
        pbar.log_message("Turn on all the layers")
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True


# In[20]:


handler = EarlyStopping(patience=6, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)


# In[21]:


checkpoints = ModelCheckpoint('models', f'Model_{model_name}_6channels', save_interval=3, n_saved=10, create_dir=True)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoints, {f'{learning_rate_str}': model})


# In[22]:


pbar = ProgressBar(bar_format='')
pbar.attach(trainer, output_transform=lambda x: {'loss': x})


# In[23]:


trainer.run(loader, max_epochs=50)

