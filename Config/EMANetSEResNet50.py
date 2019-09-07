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


class EMANetSEResNet50_768_Fold1_NIH_ptx(Config):
    name = 'MANetResNet101_768'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=EMANetSEResNet50_v2,
        pretrained=True,
        # weights=None,
        weights='./Saves/EMANetSEResNet50_v2/Fold1/11Aug_21:58/2019-08-11 23:19_Fold1_Epoch11_reset0_dice0.587_ptx_dice0.587',
    )

    image = AttributeDict(
        tgt_size=(768, 768),
        aug=Aug.Aug2a,
        mixup=0.0,
        batch_size=8,
        workers=12,
    )

    loss = losses.BCE_Lovasz(alpha=0.002)

    optim = AttributeDict(
        type='Adam',
        momentum=0.9,
        weight_decay=0.0001,
    )

    fold = 1
    train = AttributeDict(
        cycle1=AttributeDict(
            lr=1e-3 / 2,
            lr_min=1e-3 / 200,
            n_epoch=0,
            scheduler='OneCycleLR',  # 'CosineAnneling',
            tmax=10,
            tmul=1,
            grad_acc=2,
            freeze_encoder=False,
            freeze_bn=False,
        ),
        cycle2=AttributeDict(
            lr=1e-3 / 4,
            lr_min=1e-3 / 200,
            n_epoch=40,
            scheduler='CosineAnneling',  # 'OneCycleLR',
            tmax=41,
            tmul=1,
            grad_acc=8,
            freeze_encoder=False,
            freeze_bn=False,
        ),
        remove_bg_only=True,
        shuffle=True,
        use_nih=True,
        epoch_per_val=1,
        cls_csv_path='/media/nvme/Datasets/Pneumothorax/CheXpert-v1.0-small/train_val_prepared.csv',
        csv_path='./Data/Folds/fold{}.csv',
        val_mode='max',
    )

    test = AttributeDict(
        csv_path='./Data/testset.csv',
        # Path to save .csv and features. Features are saved "out_path/features/"
        out_path='./Output',
        # Save gallery features to disk every X batches
        TTA=[TTA.Nothing(mode='sigmoid'), TTA.ScaleTTA(1.333333333, mode='sigmoid')],
    )

    def __init__(self):
        super().__init__()


class EMANetSEResNet50_768_Fold5_NIH_ptx(EMANetSEResNet50_768_Fold1_NIH_ptx):
    # Model parameters
    model = AttributeDict(
        architecture=EMANetSEResNet50_v2,
        pretrained=True,
        # weights=None,
        weights='./Saves/EMANetSEResNet50_v2/Fold5/12Aug_07:27/2019-08-12 09:10_Fold5_Epoch14_reset0_dice0.583_ptx_dice0.583',
    )

    fold = 5

    def __init__(self):
        Config.__init__(self)
