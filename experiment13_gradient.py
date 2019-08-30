
# coding: utf-8

# ## Load libraries

# In[1]:


get_ipython().system('pip install -q -r requirements.txt')


# In[2]:


get_ipython().system('mkdir -p /artifacts')


# In[3]:


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

import random

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet, utils as enet_utils

from scripts.evaluate import eval_model
from scripts.transforms import gen_transform_train, gen_transform_validation
from scripts.plates_leak import apply_plates_leak

import warnings
warnings.filterwarnings('ignore')


# ## Define dataset and model

# In[4]:


img_dir = '/storage/rxrxai'
path_data = '/storage/rxrxaicsv'
device = 'cuda'
batch_size = 10
torch.manual_seed(0)
model_name = 'efficientnet-b4'
init_lr = 3e-4
end_lr = 1e-7


# In[5]:


class ImagesDS(D.Dataset):
    transform_validation = gen_transform_validation(crop_size=448)
    
    def __init__(self, df, img_dir=img_dir, mode='train', validation=False, channels=[1,2,3,4,5,6]):
        self.records = df.to_records(index=False)
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.validation = validation
        self.channels = channels

    def _get_img_path(self, index, channel, site):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{site}_w{channel}.png'])
        
    @staticmethod
    def _load_img_as_tensor(file_name, transform):
        with Image.open(file_name) as img:
            return transform(img)
        
    def __getitem__(self, index):
        transform1 = ImagesDS.transform_validation if self.validation else gen_transform_train()
        transform2 = ImagesDS.transform_validation if self.validation else gen_transform_train()
        
        paths1 = [self._get_img_path(index, ch, 1) for ch in self.channels]
        paths2 = [self._get_img_path(index, ch, 2) for ch in self.channels]
        
        img1 = torch.cat([self._load_img_as_tensor(img_path, transform1) for img_path in paths1])
        img2 = torch.cat([self._load_img_as_tensor(img_path, transform2) for img_path in paths2])
        
        if random.random() > 0.5 and not self.validation:
            img1, img2 = img2, img1
        
        if self.mode == 'train':
            return img1, img2, int(self.records[index].sirna)
        else:
            return img1, img2, self.records[index].id_code
    
    def __len__(self):
        return self.len


# In[6]:


# dataframes for training, cross-validation, and testing
df = pd.read_csv(path_data+'/train.csv')
df_train, df_val = train_test_split(df, test_size = 0.1, random_state=42)
df_test = pd.read_csv(path_data+'/test.csv')

# pytorch training dataset & loader
ds = ImagesDS(df_train, mode='train', validation=False)
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

# pytorch cross-validation dataset & loader
ds_val = ImagesDS(df_val, mode='train', validation=True)
val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4)

# pytorch test dataset & loader
ds_test = ImagesDS(df_test, mode='test', validation=True)
tloader = D.DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=4)


# In[7]:


class EfficientNetTwoInputs(nn.Module):
    def __init__(self):
        super(EfficientNetTwoInputs, self).__init__()
        self.classes = 1108
        
        model = EfficientNet.from_pretrained(model_name, num_classes=1108) 
        num_ftrs = model._fc.in_features
        model._fc = nn.Identity()
        
        # accept 6 channels
        trained_kernel = model._conv_stem.weight
        new_conv = enet_utils.Conv2dStaticSamePadding(6, 48, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=512)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        model._conv_stem = new_conv
        
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
model.load_state_dict(torch.load('./enet4/Model_efficientnet-b4_57.pth'))
model.train()


# In[8]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)


# In[9]:


metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


# #### EarlyStopping

# In[10]:


# handler = EarlyStopping(patience=30, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
# val_evaluator.add_event_handler(Events.COMPLETED, handler)


# #### LR Scheduler

# In[11]:


scheduler = CosineAnnealingScheduler(optimizer, 'lr', init_lr, end_lr, len(loader))
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

@trainer.on(Events.ITERATION_COMPLETED)
def print_lr(engine):
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    
    if epoch < 2 and iteration % 100 == 0:
        print(f'Iteration {iteration} | LR {optimizer.param_groups[0]["lr"]}')


# #### Compute and display metrics

# In[12]:


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {} | Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch, metrics['loss'], metrics['accuracy']))


# #### Save best epoch only

# In[13]:


get_ipython().system('mkdir -p models')


# In[14]:


def get_saved_model_path(epoch):
    return f'/artifacts/Model_{model_name}_{epoch + 48}.pth'

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
        print(f'\nEpoch: {best_epoch} - New best accuracy! Accuracy: {best_acc}\n\n\n')
        torch.save(model.state_dict(), best_epoch_file)


# #### Progress bar - uncomment when testing in notebook

# In[15]:


# pbar = ProgressBar(bar_format='')
# pbar.attach(trainer, output_transform=lambda x: {'loss': x})


# #### Train

# In[16]:


print('Training started\n')
trainer.run(loader, max_epochs=10)


# #### Evaluate

# In[ ]:


all_preds, _ = eval_model(model, tloader, best_epoch_file, path_data, sub_file='/artifacts/submission')


# In[ ]:


apply_plates_leak(all_preds, sub_file='/artifacts/submission_plates_leak.csv')

