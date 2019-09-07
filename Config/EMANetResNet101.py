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


class EMANetResNet101_768_Fold7(Config):
    name = 'MANetResNet101_768'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=EMANetResNet101_v2,
        pretrained=True,
        weights='./Saves/EMANetResNet101_v2/Fold7/14Aug_23:07/2019-08-15 '
                '13:15_Fold7_Epoch39_reset0_dice0.831_ptx_dice0.402',  # holdout850
    )

    image = AttributeDict(
        tgt_size=(768, 768),
        aug=Aug.Aug2a,
        mixup=0,
        batch_size=8,
        workers=12,
    )

    loss = nn.BCEWithLogitsLoss(reduction='none')

    optim = AttributeDict(
        type='Adam',
        momentum=0.9,
        weight_decay=0.0001,
    )

    fold = 7
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
        remove_bg_only=False,
        shuffle=True,
        use_nih=False,
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


class EMANetResNet101_768_Fold8(EMANetResNet101_768_Fold7):
    name = 'MANetResNet101_768'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=EMANetResNet101_v2,
        pretrained=True,
        weights='./Saves/EMANetResNet101_v2/Fold8/16Aug_00:00/2019-08-16 '
                '10:36_Fold8_Epoch29_reset0_dice0.836_ptx_dice0.479'  # holdout854
    )

    fold = 8

    def __init__(self):
        Config.__init__(self)


class EMANetResNet101_768_Fold0_ptx(EMANetResNet101_768_Fold7):
    name = 'MANetResNet101_768'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=EMANetResNet101_v2,
        pretrained=True,
        # weights=None,
        weights='./Saves/EMANetResNet101_v2/Fold0/10Aug_09:46/2019-08-10 10:59_Fold0_Epoch8_reset0_dice0.353_ptx_dice0.583',
    )

    loss = losses.BCE_Lovasz(alpha=0.01)

    fold = 0
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
        use_nih=False,
        epoch_per_val=1,
        cls_csv_path='/media/nvme/Datasets/Pneumothorax/CheXpert-v1.0-small/train_val_prepared.csv',
        csv_path='./Data/Folds/fold{}.csv',
        val_mode='max',
    )

    def __init__(self):
        Config.__init__(self)


class EMANetResNet101_768_Fold0_NIH_ptx(EMANetResNet101_768_Fold0_ptx):
    # Model parameters
    model = AttributeDict(
        architecture=EMANetResNet101_v2,
        pretrained=True,
        # weights=None,
        weights='./Saves/EMANetResNet101_v2/Fold0/10Aug_22:51/2019-08-10 23:13_Fold0_Epoch2_reset0_dice0.591_ptx_dice0.591',
    )

    fold = 0
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

    def __init__(self):
        super().__init__()


class EMANetResNet101_768_Fold1_NIH_ptx(EMANetResNet101_768_Fold0_NIH_ptx):
    # Model parameters
    model = AttributeDict(
        architecture=EMANetResNet101_v2,
        pretrained=True,
        # weights=None,
        weights='./Saves/EMANetResNet101_v2/Fold1/11Aug_08:46/2019-08-11 11:01_Fold1_Epoch12_reset0_dice0.591_ptx_dice0.591',
    )
    fold = 1

    def __init__(self):
        Config.__init__(self)
