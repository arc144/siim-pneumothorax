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


class PANetResNet50_768_Fold0(Config):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold0/29Jul_08:45/2019-07-29 19:12_Fold0_Epoch39_reset0_val0.840',
    )

    image = AttributeDict(
        tgt_size=(768, 768),
        aug=Aug.Aug2a,
        mixup=0,
        batch_size=12,
        workers=12,
    )

    loss = nn.BCEWithLogitsLoss(reduction='none')

    optim = AttributeDict(
        type='Adam',
        momentum=0.9,
        weight_decay=0.0001,
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
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=5,
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


class PANetResNet50_768_Fold1(PANetResNet50_768_Fold0):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold1/28Jul_23:16/2019-07-29 08:21_Fold1_Epoch34_reset0_val0.842',
    )

    fold = 1

    def __init__(self):
        Config.__init__(self)


class PANetResNet50_768_Fold2(PANetResNet50_768_Fold0):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold2/28Jul_21:21/2019-07-28 22:58_Fold2_Epoch6_reset0_val0.842',
    )

    fold = 2

    def __init__(self):
        Config.__init__(self)


class PANetResNet50_768_Fold3(PANetResNet50_768_Fold0):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold3/28Jul_01:57/2019-07-28 09:59_Fold3_Epoch30_reset0_val0.854',
    )

    fold = 3

    def __init__(self):
        Config.__init__(self)


class PANetResNet50_768_Fold4_ptx(PANetResNet50_768_Fold0):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold4/03Aug_00:27/2019-08-03 03:55_Fold4_Epoch38_reset0_dice0.481_ptx_dice0.520',
    )

    loss = losses.BCE_Lovasz(alpha=0.002)

    fold = 4
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
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=5,
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


class PANetResNet50_768_Fold5_ptx(PANetResNet50_768_Fold4_ptx):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold5/03Aug_04:06/2019-08-03 08:16_Fold5_Epoch37_reset0_dice0.485_ptx_dice0.565',
    )

    fold = 5

    def __init__(self):
        Config.__init__(self)


class PANetResNet50_768_Fold6_ptx(PANetResNet50_768_Fold4_ptx):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold6/03Aug_10:49/2019-08-03 13:33_Fold6_Epoch30_reset0_dice0.343_ptx_dice0.554',
    )

    fold = 6

    def __init__(self):
        Config.__init__(self)


class PANetResNet50_768_Fold4_NIH_ptx(PANetResNet50_768_Fold4_ptx):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        # weights=None,
        weights='./Saves/PANetResNet50/Fold4/12Aug_16:50/2019-08-12 19:21_Fold4_Epoch20_reset0_dice0.562_ptx_dice0.562',
    )

    fold = 4
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
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=5,
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
        Config.__init__(self)
