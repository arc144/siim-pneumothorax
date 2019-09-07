import os

import gc

from Config import *
from Core.dataset import DatasetFactory
from Core.tasks import Segmentation

save_path = './Data/Preds'
##############################################################################################

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
]

for cfg in configs:
    print('TTA: {}'.format(cfg.test.TTA))

for j, cfg in enumerate(configs):
    dataset = DatasetFactory(
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

    pred_path = os.path.join(save_path, type(cfg).__name__)
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

    if not j:
        index_vec, meta_vec, mean_pred, _ = trainer.predict(test_loader,
                                                            cfg.test.TTA,
                                                            pbar=True,
                                                            raw=True,
                                                            pred_zip=os.path.join(pred_path, 'pred_tta.zip'),
                                                            tgt_size=1024)
    else:
        mean_pred = (j * mean_pred + trainer.predict(test_loader,
                                                     cfg.test.TTA,
                                                     pbar=True,
                                                     raw=True,
                                                     pred_zip=os.path.join(pred_path, 'pred_tta.zip'),
                                                     tgt_size=1024)[2]) / (j + 1)
    gc.collect()

