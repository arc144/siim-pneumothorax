from Config import *
from Core.dataset import DatasetFactory
from Core.tasks import Segmentation
from Utils.postprocessing import count_connected_instances
import os
import numpy as np
import cv2
import pandas as pd
import gc

##############################################################################################
# cfg = config.Config()
trainset = '/media/hdd/Kaggle/Pneumothorax/Data/trainset.csv'
holdout = '/media/hdd/Kaggle/Pneumothorax/Data/Folds/holdout.csv'

from Config import PANetDilatedResNet34_768_Fold0

configs = [PANetDilatedResNet34_768_Fold0(),
           PANetDilatedResNet34_768_Fold1(),
           PANetResNet50_768_Fold0(),
           PANetResNet50_768_Fold1(),
           PANetResNet50_768_Fold2(),
           PANetResNet50_768_Fold3(),
           ]
# cfg = PANetDilatedResNet34_768_Fold0()
for i, cfg in enumerate(configs):
    dataset = DatasetFactory(
        # holdout,
        # cfg.train.csv_path,
        # trainset,
        cfg.test.csv_path,
        cfg)
    test_loader = dataset.yield_loader(is_test=True)

    net = cfg.model.architecture(pretrained=cfg.model.pretrained)

    trainer = Segmentation(net,
                           mode='test',
                           debug=False,
                           fold=cfg.fold)

    assert cfg.model.weights is not None, 'Weights is None!!'
    trainer.load_model(cfg.model.weights)
    index_vec, meta_vec, pred_vec, mask_vec = trainer.predict(test_loader,
                                                              cfg.test.TTA,
                                                              pbar=True,
                                                              raw=True,
                                                              tgt_size=1024)
    gc.collect()

    BIN_THS = 0.2
    IMSIZE = 1024

    pred_count_vec = count_connected_instances(pred_vec, ths=0.2)

    df_dict = dict(
        SoftArea=[],
        Area=[],
        NInstances=[],
        PatientID=[],
        ViewPosition=[],
        PatientAge=[],
        PatientSex=[],
        Target=[],
    )

    for i, (meta, pred, count, mask) in enumerate(zip(meta_vec, pred_vec, pred_count_vec, mask_vec)):
        df_dict['SoftArea'].append(pred.sum() / (IMSIZE ** 2))
        df_dict['Area'].append((pred > 0.2).sum() / (IMSIZE ** 2))
        df_dict['NInstances'].append(count)

        df_dict['PatientID'].append(meta['id'])
        df_dict['ViewPosition'].append(meta['view'])
        df_dict['PatientAge'].append(meta['age'])
        df_dict['PatientSex'].append(meta['sex'])
        df_dict['Target'].append(mask.sum() > 0)

    df = pd.DataFrame(df_dict)
    df.to_csv('{}_testset_features.csv'.format(type(cfg).__name__), index=False)
    del index_vec, meta_vec, pred_vec, mask_vec
    gc.collect()
