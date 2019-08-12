#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[1]:


get_ipython().system('pip install -q -r requirements.txt')


# In[1]:


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
from ignite.contrib.handlers import LinearCyclicalScheduler
from ignite.handlers import  EarlyStopping, ModelCheckpoint

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from scripts.device import get_device

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


learning_rate_str, model_name = ['35e-5', 'efficientnet-b4']
learning_rate = float(learning_rate_str)

print(f'learning rate: {learning_rate}')
print(f'model name: {model_name}')


# ## Define dataset and model

# In[3]:


img_dir = '../input/rxrxairgb512'
path_data = '../input/rxrxaicsv'
device = get_device()
batch_size = 4
torch.manual_seed(0)
print(device)


# In[4]:


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir=img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):
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

    def _get_img_path(self, index):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return f'{self.img_dir}/{self.mode}/{experiment}_{plate}_{well}_s{self.site}.jpeg'
        
    def __getitem__(self, index):
        img = self._load_img_as_tensor(self._get_img_path(index))
        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len


# In[5]:


# dataframes for training, cross-validation, and testing
df = pd.read_csv(path_data+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.05, random_state=42)
df_test = pd.read_csv(path_data+'/test.csv')

# pytorch training dataset & loader
ds = ImagesDS(df_train, mode='train')
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

# pytorch cross-validation dataset & loader
ds_val = ImagesDS(df_val, mode='train')
val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4)

# pytorch test dataset & loader
ds_test = ImagesDS(df_test, mode='test')
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)


# In[6]:


classes = 1108
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=classes) 


# In[7]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[8]:


metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


# In[10]:


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {} | LR: {:.4f}  Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch, optimizer.param_groups[0]['lr'], metrics['loss'], metrics['accuracy']))


# In[11]:


# lr_scheduler = ExponentialLR(optimizer, gamma=0.99)

# @trainer.on(Events.EPOCH_COMPLETED)
# def update_lr_scheduler(engine):
#     lr_scheduler.step()
#     lr = float(optimizer.param_groups[0]['lr'])
#     print("Learning rate: {}".format(lr))

scheduler = LinearCyclicalScheduler(optimizer, 'lr', 35e-5, 15e-5, len(loader))
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)


# In[12]:


# @trainer.on(Events.EPOCH_STARTED)
# def turn_on_layers(engine):
#     epoch = engine.state.epoch
#     if epoch == 1:
#         for name, child in model.named_children():
#             if name == '_fc':
#                 pbar.log_message(name + ' is unfrozen')
#                 for param in child.parameters():
#                     param.requires_grad = True
#             else:
#                 pbar.log_message(name + ' is frozen')
#                 for param in child.parameters():
#                     param.requires_grad = False
#     if epoch == 3:
#         pbar.log_message("Turn on all the layers")
#         for name, child in model.named_children():
#             for param in child.parameters():
#                 param.requires_grad = True


# In[13]:


handler = EarlyStopping(patience=6, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)


# In[14]:


checkpoints = ModelCheckpoint('models', f'Model_{model_name}_3channels', save_interval=2, n_saved=10, create_dir=True)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoints, {f'{learning_rate_str}': model})


# In[15]:


# pbar = ProgressBar(bar_format='')
# pbar.attach(trainer, output_transform=lambda x: {'loss': x})


# In[16]:


print('Training started')
trainer.run(loader, max_epochs=50)

