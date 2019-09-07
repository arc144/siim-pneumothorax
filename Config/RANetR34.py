import os

import imgaug as ia
import numpy as np
import torch

import Utils.TTA as TTA
import Utils.augmentations as Aug
import Backbone as B
from Core import losses
from Models.models import *
from .UNet import Config


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


class RANetR34_768_Fold0(Config):
    name = 'RANetR34_768'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=RANetR34,
        pretrained=True,
        weights='/media/hdd/Kaggle/Pneumothorax/Saves/RANetR34/Fold0/20Aug_00:32/2019-08-20 '
                '05:37_Fold0_Epoch35_reset2_dice0.810_ptx_dice0.440', #holdout846

    )

    image = AttributeDict(
        tgt_size=(768, 768),
        aug=Aug.Aug2a,
        mixup=0.2,
        batch_size=16,
        workers=6,
    )
    fold = 0

    test = AttributeDict(
        csv_path='/media/hdd/Kaggle/Pneumothorax/Data/testset.csv',
        # Path to save .csv and features. Features are saved "out_path/features/"
        out_path='/media/hdd/Kaggle/Pneumothorax/Output',
        # Save gallery features to disk every X batches
        TTA=[TTA.Nothing(mode='sigmoid')],
        # TTA=[TTA.Nothing(mode='sigmoid'), TTA.ScaleTTA(1.333333333, mode='sigmoid')],
        # TTA=[TTA.Nothing(), TTA.ScaleTTA(scale=np.sqrt(2)), TTA.ScaleTTA(scale=1 / np.sqrt(2))],  # TTA.VFlip(),
    )
    def __init__(self):
        super().__init__()


class RANetR34_768_Fold0b(RANetR34_768_Fold0):
    fold = 0
    model = AttributeDict(
        architecture=RANetR34,
        pretrained=True,
        weights='/media/hdd/Kaggle/Pneumothorax/Saves/RANetR34/Fold0/20Aug_00:32/2019-08-20 '
                '02:43_Fold0_Epoch15_reset1_dice0.804_ptx_dice0.434', #holdout
    )