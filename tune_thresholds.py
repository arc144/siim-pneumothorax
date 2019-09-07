from Config import *
from Core.dataset import DatasetFactory
from Core.tasks import Segmentation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Core import metrics
from Utils import TTA


def do_validation(prediction, ground_truth, ths=0.2, cls_ths=0.6, noise_th=75.0 * (384 / 128.0) ** 2):
    pred = prediction.copy()
    gt = ground_truth.copy()

    if noise_th is not None:
        pred[pred.reshape(pred.shape[0], -1).sum(-1) < noise_th, ...] = 0.0
    pred_seg = (pred > ths)
    if cls_ths is not None:
        pred_cls = (pred > cls_ths)
        pred_seg[pred_cls.reshape(pred.shape[0], -1).sum(1) == 0] = 0.0
    # pred[pred.reshape(pred.shape[0], -1).sum(-1) < noise_th, ...] = 0.0
    dice, ptx_dice = metrics.cmp_dice(pred_seg, gt)
    acc = metrics.cmp_cls_acc(pred_seg, gt).mean()
    print('Dice: {:4f}\tPtx_Dice: {:4f}\tAcc: {:4f}'.format(dice.mean(), ptx_dice.mean(), acc))
    return dice


##############################################################################################
holdout = '/media/hdd/Kaggle/Pneumothorax/Data/Folds/holdout.csv'
configs = [
    # all images
    PANetDilatedResNet34_768_Fold0(),
    PANetDilatedResNet34_768_Fold1(),
    PANetDilatedResNet34_768_Fold3(),
    PANetDilatedResNet34_768_Fold4(),
    PANetResNet50_768_Fold0(),
    PANetResNet50_768_Fold1(),
    PANetResNet50_768_Fold2(),
    PANetResNet50_768_Fold3(),
    EMANetResNet101_768_Fold7(),
    EMANetResNet101_768_Fold8(),
    # ptx only
    PANetDilatedResNet34_768_Fold5_NIH_ptx(),
    PANetDilatedResNet34_768_Fold6_NIH_ptx(),
    PANetDilatedResNet34_768_Fold7_NIH_ptx(),
    PANetResNet50_768_Fold4_NIH_ptx(),
    PANetResNet50_768_Fold5_ptx(),
    PANetResNet50_768_Fold6_ptx(),
    EMANetResNet101_768_Fold0_NIH_ptx(),
    EMANetResNet101_768_Fold1_NIH_ptx(),
    EMANetSEResNet50_768_Fold1_NIH_ptx(),
    EMANetSEResNet50_768_Fold5_NIH_ptx(),
]

for j, cfg in enumerate(configs):
    dataset = DatasetFactory(
        holdout,
        # cfg.train.csv_path,
        cfg)
    val_loader = dataset.yield_loader(is_test=True)
    net = cfg.model.architecture(pretrained=cfg.model.pretrained)

    trainer = Segmentation(net,
                           mode='test',
                           criterion=cfg.loss,
                           debug=False,
                           fold=cfg.fold)

    assert cfg.model.weights is not None, 'Weights is None!!'
    trainer.load_model(cfg.model.weights)
    _, _, pred, mask = trainer.predict(val_loader, cfg.test.TTA,
                                       raw=True, tgt_size=768, pbar=True)
    if not j:
        mean_pred = pred
    else:
        mean_pred = (j * mean_pred + pred) / (j + 1)

thresholds = np.arange(0.1, 0.8, 0.1)
min_sizes = np.arange(30, 290, 20) * (384 / 128) ** 2
cls_thresholds = np.arange(0.6, 1, 0.1)
# min_sizes = np.arange(190, 270, 20) * (384 / 128) ** 2

scores_ths = []
scores_minsz = []

# for i, t in enumerate(tqdm(thresholds)):
#     print('Threshold: {:.3f}'.format(t))
#     scores_ths.append(do_validation(mean_pred, mask, ths=t, noise_th=None, cls_ths=None))

# for i, t in enumerate(tqdm(thresholds)):
#     print('Threshold: {:.3f}'.format(t))
#     scores_ths.append(do_validation(mean_pred, mask, ths=t, cls_ths=None))

for i, msz in enumerate(tqdm(min_sizes)):
    print('Noise threshold: {:.3f}'.format(msz))
    scores_minsz.append(do_validation(mean_pred, mask, ths=0.2, noise_th=msz, cls_ths=None))

# for i, cls_t in enumerate(tqdm(cls_thresholds)):
#     print('Noise threshold: {:.3f}'.format(cls_t))
#     scores_minsz.append(do_validation(mean_pred, mask, ths=0.2, noise_th=None, cls_ths=cls_t))

fig, axs = plt.subplots(1, 2)
axs[0].plot(thresholds, scores_ths)
axs[1].plot(min_sizes, scores_minsz)
plt.show()
print('end')
