import Utils.TTA as TTA
import Utils.augmentations as Aug
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
        weights='./Saves/PANetDilatedResNet34/Fold0/29Aug_22:13/2019-08-30 06:00_Fold0_Epoch32_reset0_dice0.846_ptx_dice0.502',
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
        TTA=[TTA.Nothing(mode='sigmoid'), TTA.ScaleTTA(1.333333333, mode='sigmoid')],
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold1(PANetDilatedResNet34_768_Fold0):
    fold = 1
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold1/30Aug_21:31/2019-08-31 '
                '06:29_Fold1_Epoch37_reset0_dice0.856_ptx_dice0.478',
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold3(PANetDilatedResNet34_768_Fold0):
    fold = 3
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold3/31Aug_07:13/2019-08-31 13:48_Fold3_Epoch27_reset0_dice0.846_ptx_dice0.463',
    )

    def __init__(self):
        super().__init__()


class PANetDilatedResNet34_768_Fold4(PANetDilatedResNet34_768_Fold0):
    fold = 4
    model = AttributeDict(
        architecture=PANetDilatedResNet34,
        pretrained=True,
        weights='./Saves/PANetDilatedResNet34/Fold4/31Aug_16:59/2019-09-01 01:35_Fold4_Epoch36_reset0_dice0.861_ptx_dice0.457',
    )

    def __init__(self):
        super().__init__()