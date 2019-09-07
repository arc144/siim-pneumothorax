import os

import imgaug as ia
import numpy as np

import Utils.TTA as TTA
import Utils.augmentations as Aug
from Models.models import *


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


class Config(object):
    name = 'BaseConfig'
    seed = 2019
    task = 'seg'

    # Model parameters
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights=None,
    )

    image = AttributeDict(
        tgt_size=(768, 768),
        aug=Aug.Aug2a,
        mixup=0.0,
        batch_size=30,
        workers=12,
    )

    loss = nn.BCEWithLogitsLoss(reduction='none')

    optim = AttributeDict(
        type='Adam',
        momentum=0.9,
        weight_decay=0.0001,
    )

    fold = 5
    train = AttributeDict(
        cycle1=AttributeDict(
            lr=1e-3 / 2,
            lr_min=1e-3 / 200,
            n_epoch=0,
            scheduler='OneCycleLR',
            tmax=10,
            tmul=1,
            grad_acc=2,
            freeze_encoder=False,
            freeze_bn=False,
        ),
        cycle2=AttributeDict(
            lr=1e-3 / 4,
            lr_min=1e-3 / 200,
            n_epoch=30,
            scheduler='CosineAnneling',
            tmax=31,
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
        val_mode='max',
    )

    test = AttributeDict(
        csv_path='/media/hdd/Kaggle/Pneumothorax/Data/testset.csv',
        # Path to save .csv and features. Features are saved "out_path/features/"
        out_path='/media/hdd/Kaggle/Pneumothorax/Output',
        # Save gallery features to disk every X batches
        TTA=[TTA.Nothing(mode='sigmoid')],
    )

    def __init__(self):
        self.train.csv_path = '/media/hdd/Kaggle/Pneumothorax/Data/Folds/fold{}.csv'.format(self.fold)

        if self.train.use_nih:
            self.train.csv_path = self.train.csv_path.replace('.csv', '_nih.csv')
        if self.task == 'cls':
            self.train.csv_path = self.train.cls_csv_path
        print('Fold {} selected'.format(self.fold))
        print(self.train.csv_path)
        # Check if paths exist
        paths = [self.train.csv_path,
                 self.test.csv_path
                 ]
        for p in paths:
            assert os.path.exists(p), "Path does not exists. Got {}".format(p)
        # Check if out_path exist and create it
        if not os.path.exists(self.test.out_path):
            os.mkdir(self.test.out_path)

        # Set random seeds
        seed = self.seed + self.fold
        ia.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _init_fn(self, worker_id):
        np.random.seed(self.seed + worker_id)
