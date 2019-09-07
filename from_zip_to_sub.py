import os
from zipfile import ZipFile

import cv2
import gc
import numpy as np
from tqdm import tqdm

from Utils.postprocessing import make_submission

fns = [

    './Data/Preds/PANetDilatedResNet34_768_Fold0/pred_tta.zip',
    './Data/Preds/PANetDilatedResNet34_768_Fold1/pred_tta.zip',
    './Data/Preds/PANetDilatedResNet34_768_Fold3/pred_tta.zip',
    './Data/Preds/PANetDilatedResNet34_768_Fold4/pred_tta.zip',

    './Data/Preds/PANetResNet50_768_Fold0/pred_tta.zip',
    './Data/Preds/PANetResNet50_768_Fold1/pred_tta.zip',
    './Data/Preds/PANetResNet50_768_Fold2/pred_tta.zip',
    './Data/Preds/PANetResNet50_768_Fold3/pred_tta.zip',

    './Data/Preds/EMANetResNet101_768_Fold7/pred_tta.zip',
    './Data/Preds/EMANetResNet101_768_Fold8/pred_tta.zip',

    './See/Model_000_f00/f00-PREDS.zip',
    './See/Model_000_f01/f01-PREDS.zip',
    './See/Model_000_f02/f02-PREDS.zip',
    './See/Model_000_f03/f03-PREDS.zip',
    './See/Model_000_f04/f04-PREDS.zip',

    './See/Model_001_f00/f00-PREDS.zip',
    './See/Model_001_f01/f01-PREDS.zip',
    './See/Model_001_f02/f02-PREDS.zip',
    './See/Model_001_f03/f03-PREDS.zip',
    './See/Model_001_f04/f04-PREDS.zip',

    './See/Model_002_f00/f00-PREDS.zip',
    './See/Model_002_f01/f01-PREDS.zip',
    './See/Model_002_f02/f02-PREDS.zip',
    './See/Model_002_f03/f03-PREDS.zip',
    './See/Model_002_f04/f04-PREDS.zip',
    './See/Model_002_f05/f05-PREDS.zip',
    './See/Model_002_f06/f06-PREDS.zip',
]

weights = np.array([1] * 27)
assert len(weights) == len(fns)

handles = [ZipFile(fn) for fn in fns]

image_ids, rles = [], []
pngs = [x.split('.')[0] for x in ZipFile(fns[0]).namelist()]
predictions = np.zeros((len(pngs), 1024, 1024), dtype=np.float16)
for i, png in enumerate(tqdm(pngs)):
    image_id = os.path.splitext(png)[0]
    p_ensemble = 0.0
    for j, (handle, w) in enumerate(zip(handles, weights)):
        # This is a fix to a minor bug that adds ".png" to saved predictions
        if j < 10:
            ext = '.dcm.png'
        else:
            ext = '.png'
        with handle.open(png + ext) as f:
            img = cv2.imdecode(np.frombuffer(f.read(), 'uint8'), 0)
            p = np.float32(img) / 255
            p_ensemble += p * w / np.sum(weights)
            predictions[i] = p_ensemble
    if i % 100 == 0:
        gc.collect()

gc.collect()

ths = 0.20,
cls_ths = 0.72

mean_pred = predictions.copy()

pred_cls = (mean_pred > cls_ths)
mean_pred[pred_cls.reshape(pred_cls.shape[0], -1).sum(1) == 0] = 0.0

mean_pred = (mean_pred > ths)
mean_pred = 255. * mean_pred

count = (mean_pred.reshape(mean_pred.shape[0], -1).sum(1) > 1).sum()
print('{:} non empty images ({:.2f}%)'.format(count, count / len(mean_pred)))
make_submission([os.path.splitext(x)[0] for x in pngs], mean_pred, os.path.join('./Output/ens_submission.csv'))
del mean_pred
gc.collect()
