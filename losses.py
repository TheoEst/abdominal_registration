# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:31:16 2020

@author: T_ESTIENNE
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math
from abdominal_registration import utils


def dice_loss(input, target):
    smooth = 1.
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(input, target):
    
    assert input.shape[1] in [12, 14]
    assert target.shape[1] in [12, 14]

    dice = 0
    n_channels = input.shape[1]
    
    for i in range(1,n_channels):
        dice += dice_loss(input[:, i, ...], target[:, i, ...])

    return dice/ (n_channels -1)

def masked_mean_dice_loss(input, target, label):

    assert input.shape[1] == 12
    assert target.shape[1] == 12

    dice = 0
    batch_size = input.shape[0]
    channel = input.shape[1]
    
    for i in range(1, channel):
        dice_patient = 0
        n_patient = 0
        for j in range(batch_size):
            if label[j, i] == 1:
                dice_patient += dice_loss(input[j, i, ...], target[j, i, ...])
                n_patient += 1

        if n_patient > 0: 
            dice += (dice_patient / n_patient)

    return dice/ (channel-1)

def dice_metrics(mask, gt):
    '''
    Computes metrics based on the confusion matrix!
    '''
    lnot = np.logical_not
    land = np.logical_and

    true_positive = np.sum(land((mask), (gt)))
    false_positive = np.sum(land((mask), lnot(gt)))
    false_negative = np.sum(land(lnot(mask), (gt)))
    true_negative = np.sum(land(lnot(mask), lnot(gt)))

    M = np.array([[true_negative, false_negative],
                  [false_positive, true_positive]]).astype(np.float64)
    metrics = {}
    metrics['Sensitivity'] = M[1, 1] / (M[0, 1] + M[1, 1])
    metrics['Specificity'] = M[0, 0] / (M[0, 0] + M[1, 0])
    metrics['Dice'] = 2 * M[1, 1] / (M[1, 1] * 2 + M[1, 0] + M[0, 1])
    # metrics may be NaN if denominator is zero! use np.nanmean() while
    # computing average to ignore NaNs.

    return metrics['Dice']
         
def evalAllSample(mask, gt, patients, args):
    '''
    1 spleen, 2 right kidney, 3 left kidney, 
    6 liver -> 4, 7 stomach -> 5, 11 pancreas -> 6

    '''
    batch_size = mask.shape[0]
    mask = utils.to_numpy(args, mask)
    gt = utils.to_numpy(args, gt)
    dice_dict = {}
    
    labels = ['Spleen', 'Right_Kidney', 'Left_Kidney', 'Gall_Bladder',
              'Esophagus', 'Liver', 'Stomach', 'Aorta', 'Inferior_Vena_Cava',
              'Portal&Splenic_Vein', 'Pancreas']
    
    if gt.shape[1] == 14:
        labels += ['Left_Adrenal_Gland', 'Right_Adrenal_Gland']

    channels = len(labels)
    
    for batch in range(batch_size):

        patient_dice = {}
        for i in range(0, channels):

            gt_ = gt[batch, i+1, ...]
            mask_ = mask[batch, i+1, ...] > 0.5
            if np.sum(gt_) == 0: # No ground truth:
                dice = math.nan
            else:
                dice = dice_metrics(mask_, gt_)
            patient_dice[labels[i]] = dice

        dice_dict[patients[batch]]  = patient_dice

    return pd.DataFrame.from_dict(dice_dict, orient='index')

