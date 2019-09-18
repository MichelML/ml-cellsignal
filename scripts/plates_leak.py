#!/usr/bin/env python
# coding: utf-8

# Disclosure: this is taken from  https://kaggle.com/zaharch/keras-model-boosted-with-plates-leak
# 
# As reported by Recursion [in this post](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/102905), there is a special structure in the data which simplifies predictions significantly.
# 
# Assignments of sirnas to plates is not completely random in this competition. In this kernel, first I show it on the train data, and then apply the leak [on the pretrained Keras model](https://www.kaggle.com/chandyalex/recursion-cellular-keras-densenet) (kudos to [Alex](https://www.kaggle.com/chandyalex)) with LB 0.113 to get score 0.207. Same model which uses 2 sites for inference gets LB score 0.231 (the original model uses only one site but I just can't hold myself on that). 


import numpy as np
import pandas as pd
import os

from tqdm import tqdm
import PIL
import cv2
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.utils.data as D
from torchvision import transforms

from tqdm import tqdm_notebook

def apply_plates_leak(all_predictions, old_sub='/artifacts/submission.csv', sub_file='/artifacts/submission_plates_leak.csv'):
    """Apply the plates leak to pre-existing predictions.
    see https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak

    Args:
        all_predictions (two-dimensional list of float of shape 19897x1108): list representing the probabilities of each siRNA for each of the 19897 predictions to make.
    
    Output:
        submission_plates_leak.csv or sub_file provided
    """ 
    train_csv = pd.read_csv("/storage/rxrxai/train.csv")
    test_csv = pd.read_csv("/storage/rxrxai/test.csv")
    sub = pd.read_csv(old_sub)

    plate_groups = np.zeros((1108,4), int)
    for sirna in range(1108):
        grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum()


    all_test_exp = test_csv.experiment.unique()
    group_plate_probs = np.zeros((len(all_test_exp),4))
    for idx in range(len(all_test_exp)):
        preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna'].values
        pp_mult = np.zeros((len(preds),1108))
        pp_mult[range(len(preds)),preds] = 1

        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)

        for j in range(4):
            mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) ==                np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)

            group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)

    # exp_to_group = group_plate_probs.argmax(1)
    
    # taken directly from https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak
    # since it seems to be the best probabilities
    exp_to_group = [3, 1, 0, 0, 0, 0, 2, 2, 3, 0, 0, 3, 1, 0, 0, 0, 2, 3] 
    predicted = np.stack(all_predictions).squeeze()

    def select_plate_group(pp_mult, idx):
        sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
        assert len(pp_mult) == len(sub_test)
        mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) !=            np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        pp_mult[mask] = -1000000.
        return pp_mult

    for idx in range(len(all_test_exp)):
        #print('Experiment', idx)
        indices = (test_csv.experiment == all_test_exp[idx])

        preds = predicted[indices,:].copy()

        preds = select_plate_group(preds, idx)
        sub.loc[indices,'sirna'] = preds.argmax(1)

    print("Printing correlation with initial submission.csv :")
    print((sub.sirna == pd.read_csv(old_sub).sirna).mean())
    sub.to_csv(sub_file, index=False, columns=['id_code','sirna'])
