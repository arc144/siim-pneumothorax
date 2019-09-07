import Utils.TTA as TTA
import Utils.augmentations as Aug
from Core import losses
from Models.models import *
from .BaseConfig import Config


class AttributeDict(dict):
    __getattr__ = dict.__getitem__


class PANetResNet50_768_Fold0(Config):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        weights='./Saves/PANetResNet50/Fold0/01Sep_02:33/2019-09-01 12:16_Fold0_Epoch36_reset0_dice0.849_ptx_dice0.525',
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
            grad_acc=5,
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


class PANetResNet50_768_Fold1(PANetResNet50_768_Fold0):
    name = 'BaseConfig'
    seed = 2019

    # Model parameters
    model = AttributeDict(
        architecture=PANetResNet50,
        pretrained=True,
        weights='./Saves/PANetResNet50/Fold1/01Sep_20:16/2019-09-02 07:02_Fold1_Epoch40_reset0_dice0.848_ptx_dice0.496',
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
        weights='./Saves/PANetResNet50/Fold2/02Sep_07:02/2019-09-02 17:02_Fold2_Epoch37_reset0_dice0.854_ptx_dice0.479',
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
        weights='./Saves/PANetResNet50/Fold3/02Sep_17:51/2019-09-03 04:06_Fold3_Epoch38_reset0_dice0.850_ptx_dice0.479',
    )

    fold = 3

    def __init__(self):
        Config.__init__(self)