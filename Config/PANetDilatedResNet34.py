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


class PANetDilatedResNet34_768_Fold0(Config):
    name = 'PANetDilatedResNet34_768'
    seed = 2019
    task = 'seg'

    # Model parameters
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold0/18Jul_14:21_v2_768/2019-07-18 '
                '17:21_Fold0_Epoch16_reset0_val0.833',  # holdout848
    )

    image = AttributeDict(
        tgt_size=(768, 768),
        aug=Aug.Aug2a,
        mixup=0,
        batch_size=30,
        workers=6,
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
            lr=1e-3,
            lr_min=1e-3 / 25,
            n_epoch=0,
            scheduler=None,  # 'OneCycleLR',
            tmax=6,
            tmul=1,
            grad_acc=1,
            freeze_encoder=True,
            freeze_bn=False,
        ),
        cycle2=AttributeDict(
            lr=1e-3 / 4,
            lr_min=1e-3 / 200,
            n_epoch=40,
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=2,
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
        # TTA=[TTA.Nothing(mode='sigmoid')],
        TTA=[TTA.Nothing(mode='sigmoid'), TTA.ScaleTTA(1.333333333, mode='sigmoid')],
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold1(PANetDilatedResNet34_768_Fold0):
    fold = 1
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold1/21Jul_22:18/2019-07-22 07:32_Fold1_Epoch49_reset0_val0.847',
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold3(PANetDilatedResNet34_768_Fold0):
    fold = 3
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold3/31Jul_00:02/2019-07-31 '
                '03:11_Fold3_Epoch16_reset0_val0.851',  # holdout853
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold4(PANetDilatedResNet34_768_Fold0):
    fold = 4
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold4/01Aug_08:49/2019-08-01 '
                '11:08_Fold4_Epoch11_reset0_val0.855',  # holdout852
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold5_ptx(PANetDilatedResNet34_768_Fold0):
    fold = 5
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold5/01Aug_12:11/2019-08-01 14:44_Fold5_Epoch37_reset0_val0.663_valptx0.555',
    )

    loss = losses.BCE_Lovasz(alpha=0.002)

    train = AttributeDict(
        cycle1=AttributeDict(
            lr=1e-3,
            lr_min=1e-3 / 25,
            n_epoch=0,
            scheduler=None,  # 'OneCycleLR',
            tmax=6,
            tmul=1,
            grad_acc=1,
            freeze_encoder=True,
            freeze_bn=False,
        ),
        cycle2=AttributeDict(
            lr=1e-3 / 4,
            lr_min=1e-3 / 200,
            n_epoch=40,
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=2,
            freeze_encoder=False,
            freeze_bn=False,
        ),
        remove_bg_only=True,
        shuffle=True,
        use_nih=False,
        epoch_per_val=1,
        cls_csv_path='/media/nvme/Datasets/Pneumothorax/CheXpert-v1.0-small/train_val_prepared.csv',
        csv_path='./Data/Folds/fold{}.csv'.format(fold),
        val_mode='max',
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold6_ptx(PANetDilatedResNet34_768_Fold5_ptx):
    fold = 6
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold6/01Aug_21:38/2019-08-01 22:05_Fold6_Epoch6_reset0_dice0.341_ptx_dice0.542',
    )
    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold7_ptx(PANetDilatedResNet34_768_Fold5_ptx):
    fold = 7
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold7/02Aug_17:50/2019-08-02 18:48_Fold7_Epoch12_reset0_dice0.210_ptx_dice0.517',
    )
    def __init__(self):
        super().__init__()

class PANetDilatedResNet34_768_Fold5_NIH_ptx(PANetDilatedResNet34_768_Fold0):
    fold = 5
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold5/12Aug_09:58/2019-08-12 10:45_Fold5_Epoch8_reset0_dice0.573_ptx_dice0.573',
    )

    loss = losses.BCE_Lovasz(alpha=0.002)

    train = AttributeDict(
        cycle1=AttributeDict(
            lr=1e-3,
            lr_min=1e-3 / 25,
            n_epoch=0,
            scheduler=None,  # 'OneCycleLR',
            tmax=6,
            tmul=1,
            grad_acc=1,
            freeze_encoder=True,
            freeze_bn=False,
        ),
        cycle2=AttributeDict(
            lr=1e-3 / 4,
            lr_min=1e-3 / 200,
            n_epoch=40,
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=2,
            freeze_encoder=False,
            freeze_bn=False,
        ),
        remove_bg_only=True,
        shuffle=True,
        use_nih=True,
        epoch_per_val=1,
        cls_csv_path='/media/nvme/Datasets/Pneumothorax/CheXpert-v1.0-small/train_val_prepared.csv',
        csv_path='./Data/Folds/fold{}.csv'.format(fold),
        val_mode='max',
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold6_NIH_ptx(PANetDilatedResNet34_768_Fold5_NIH_ptx):
    fold = 6
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold6/12Aug_12:21/2019-08-12 13:09_Fold6_Epoch8_reset0_dice0.565_ptx_dice0.565',
    )
    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold7_NIH_ptx(PANetDilatedResNet34_768_Fold5_NIH_ptx):
    fold = 7
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold7/12Aug_14:47/2019-08-12 15:53_Fold7_Epoch11_reset0_dice0.554_ptx_dice0.554',
    )
    def __init__(self):
        super().__init__()
