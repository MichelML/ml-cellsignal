
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

from torchvision import models, transforms

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

from scripts.evaluate import eval_model 

import warnings
warnings.filterwarnings('ignore')


# ## Define dataset and model

# In[2]:


img_dir = '../input/rxrxairgb'
path_data = '../input/rxrxaicsv'
device = 'cuda'
batch_size = 256
torch.manual_seed(0)
learning_rate = 0.1
model_name = 'resnet18'


# In[3]:


class ImagesDS(D.Dataset):
    def __init__(self, df, img_dir=img_dir, mode='train', site=1):
        self.records = df.to_records(index=False)
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


# In[4]:


# dataframes for training, cross-validation, and testing
df = pd.read_csv(path_data+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
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


# In[5]:


classes = 1108

model = getattr(models, model_name)(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, classes)


# In[6]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[7]:


metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


# In[8]:


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {} | LR: {:.4f}  Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch, optimizer.param_groups[0]['lr'], metrics['loss'], metrics['accuracy']))


# In[9]:


get_ipython().system('mkdir -p models')


# In[10]:


def get_saved_model_path(epoch):
    return f'models/Model_{model_name}_{epoch}.pth'

best_acc = 0.
best_epoch = 1
best_epoch_file = ''

@trainer.on(Events.EPOCH_COMPLETED)
def save_best_epoch_only(engine):
    epoch = engine.state.epoch

    global best_acc
    global best_epoch
    global best_epoch_file
    best_acc = 0. if epoch == 1 else best_acc
    best_epoch = 1 if epoch == 1 else best_epoch
    best_epoch_file = '' if epoch == 1 else best_epoch_file

    metrics = val_evaluator.run(val_loader).metrics

    if metrics['accuracy'] > best_acc:
        prev_best_epoch_file = get_saved_model_path(best_epoch)
        if os.path.exists(prev_best_epoch_file):
            os.remove(prev_best_epoch_file)
            
        best_acc = metrics['accuracy']
        best_epoch = epoch
        best_epoch_file = get_saved_model_path(best_epoch)
        print(f'\nEpoch: {best_epoch} - New best accuracy! Accuracy: {best_acc}')
        torch.save(model.state_dict(), best_epoch_file)


# In[11]:


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate

    if epoch > 30:
        lr = learning_rate / 10.
    if epoch > 60:
        lr = learning_rate / 100.
    if epoch > 90:
        lr = learning_rate / 1000.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@trainer.on(Events.EPOCH_COMPLETED)
def update_lr_scheduler(engine):
    adjust_learning_rate(optimizer, engine.state.epoch)
    lr = float(optimizer.param_groups[0]['lr'])


# In[13]:


print('Training started')
trainer.run(loader, max_epochs=120)


# In[14]:


eval_model(model, tloader, best_epoch_file, path_data)

