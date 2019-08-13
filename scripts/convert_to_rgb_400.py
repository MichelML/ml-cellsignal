import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rxrx.io as rio
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
from tqdm import tqdm


data_path = './data'
train_df = pd.read_csv(f'{data_path}/train.csv')
test_df = pd.read_csv(f'{data_path}/test.csv')

def convert_to_rgb(df, split, resize=True, new_size=400, extension='jpeg'):
    N = df.shape[0]

    for i in tqdm(range(N)):
        code = df['id_code'][i]
        experiment = df['experiment'][i]
        plate = df['plate'][i]
        well = df['well'][i]

        for site in [1, 2]:
            save_path = f'rxrxrgb/{split}/{code}_s{site}.{extension}'

            im = rio.load_site_as_rgb(
                    split, experiment, plate, well, site, 
                    base_path=data_path
                    )
            im = im.astype(np.uint8)
            im = Image.fromarray(im)

            if resize:
                im = im.resize((new_size, new_size), resample=Image.BILINEAR)
            im.save(save_path)


if __name__ == '__main__':
    convert_to_rgb(train_df, 'train')
    convert_to_rgb(test_df, 'test')
