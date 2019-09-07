import Utils.TTA as TTA
import Utils.augmentations as Aug
from Core import losses
from Models.models import *
from .BaseConfig import Config


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


class EMANetResNet101_768_Fold7(Config):
    name = 'MANetResNet101_768'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=EMANetResNet101_v2,
        pretrained=True,
        weights='./Saves/EMANetResNet101_v2/Fold7/03Sep_18:22/2019-09-04 00:04_Fold7_Epoch37_reset0_dice0.837_ptx_dice0.412',
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
            n_epoch=40,
            scheduler='CosineAnneling',
            tmax=41,
            tmul=1,
            grad_acc=8,
            freeze_encoder=False,
            freeze_bn=False,
        ),
        remove_bg_only=False,
        shuffle=True,
        epoch_per_val=1,
        val_mode='max',
    )

    test = AttributeDict(
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
        weights='./Saves/EMANetResNet101_v2/Fold8/04Sep_00:50/2019-09-04 14:07_Fold8_Epoch35_reset0_dice0.842_ptx_dice0.444'
    )

    fold = 8

    def __init__(self):
        Config.__init__(self)