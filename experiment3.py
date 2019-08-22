
# coding: utf-8

# ## Load libraries

# In[1]:


get_ipython().system('pip install -q -r requirements.txt')


# In[ ]:


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

from torchvision import models, transforms

from ignite.engine import Events
from scripts.ignite import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

from scripts.evaluate import eval_model 

import warnings
warnings.filterwarnings('ignore')


# ## Define dataset and model

# In[ ]:


img_dir = '../input/rxrxairgb'
path_data = '../input/rxrxaicsv'
device = 'cuda'
batch_size = 200
torch.manual_seed(0)
model_name = 'resnet18'


# In[ ]:


jitter = (0.6, 1.4)
class ImagesDS(D.Dataset):
    # taken textbook from https://arxiv.org/pdf/1812.01187.pdf
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=.1),
        transforms.RandomHorizontalFlip(p=0.5),
        # PCA Noise should go here,
        transforms.ToTensor(),
        transforms.Normalize(mean=(123.68, 116.779, 103.939), std=(58.393, 57.12, 57.375))
    ])
    
    transform_validation = transforms.Compose([
        transforms.CenterCrop(224),
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


# In[ ]:


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
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)


# In[ ]:


class ResNetTwoInputs(nn.Module):
    def __init__(self):
        super(ResNetTwoInputs, self).__init__()
        self.classes = 1108
        
        model = getattr(models, model_name)(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Identity()
        
        self.resnet = model
        self.avgpool2d = nn.AvgPool2d(3)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, self.classes)

    def forward(self, x1, x2):
        x1_out = self.resnet(x1)
        x2_out = self.resnet(x2)
   
        N, _, _, _ = x1.size()
        x1_out = x1_out.view(N, -1)
        x2_out = x2_out.view(N, -1)
        
        out = torch.cat((x1_out, x2_out), 1)
        
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out 
    
model = ResNetTwoInputs()


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[ ]:


metrics = {
    'loss': Loss(criterion),
    'accuracy': Accuracy(),
}

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


# #### EarlyStopping

# In[ ]:


handler = EarlyStopping(patience=50, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
val_evaluator.add_event_handler(Events.COMPLETED, handler)


# #### LR Scheduler

# In[ ]:


scheduler = CosineAnnealingScheduler(optimizer, 'lr', 3e-4, 1e-7, len(loader))
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

@trainer.on(Events.ITERATION_COMPLETED)
def print_lr(engine):
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    
    if epoch < 3 and iteration % 10 == 0:
        print(f'Iteration {iteration} | LR {optimizer.param_groups[0]["lr"]}')


# #### Compute and display metrics

# In[ ]:


@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {} | Average Loss: {:.4f} | Accuracy: {:.4f} "
          .format(engine.state.epoch, metrics['loss'], metrics['accuracy']))


# #### Save best epoch only

# In[ ]:


get_ipython().system('mkdir -p models')


# In[ ]:


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
        print(f'\nEpoch: {best_epoch} - New best accuracy! Accuracy: {best_acc}\n\n\n')
        torch.save(model.state_dict(), best_epoch_file)


# #### Progress bar - uncomment when testing in notebook

# In[ ]:


# pbar = ProgressBar(bar_format='')
# pbar.attach(trainer, output_transform=lambda x: {'loss': x})


# #### Train

# In[ ]:


print('Training started\n')
trainer.run(loader, max_epochs=120)


# #### Evaluate

# In[ ]:


eval_model(model, tloader, best_epoch_file, path_data)

