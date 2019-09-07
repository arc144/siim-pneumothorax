import Config.UNet as config
from Core.dataset import DatasetFactory
from Core.tasks import Segmentation
from Utils.postprocessing import make_submission
import os

##############################################################################################
# cfg = config.Config()
from Config import PANetDilatedResNet34_768_Fold0
cfg = PANetDilatedResNet34_768_Fold0()

dataset = DatasetFactory(cfg.test.csv_path, cfg)
test_loader = dataset.yield_loader(is_test=True)

# Print params
print('#####################################')
print('NET: {}'.format(cfg.model.architecture.__name__))
print('TGT_SIZE: {}'.format(cfg.image.tgt_size))
print('BATCH_SIZE: {}'.format(cfg.image.batch_size))
print('#####################################')

##########################################################################################
#################################### TRAINING ############################################
##########################################################################################
net = cfg.model.architecture(pretrained=cfg.model.pretrained)

trainer = Segmentation(net,
                       mode='test',
                       debug=False,
                       fold=cfg.fold)

assert cfg.model.weights is not None, 'Weights is None!!'
trainer.load_model(cfg.model.weights)
index_vec, meta_vec, pred_vec, _ = trainer.predict(test_loader,
                                                   cfg.test.TTA,
                                                   pbar=True,
                                                   ths=0.2,
                                                   noise_th=2250) # 8526
print('{} non empty images'.format((pred_vec.reshape(pred_vec.shape[0], -1).sum(1) > 1).sum()))
make_submission(index_vec, pred_vec, os.path.join(cfg.test.out_path, 'sub.csv'))

# import matplotlib.pyplot as plt
# import numpy as np
# fig, axs = plt.subplots(40, 3, figsize=(20, 40))
# axs = axs.ravel()
# for i in range(120):
#     axs[i].imshow(np.sum(
#         [(j + 1) * (255 / len(pred_vec[i])) * pred_vec[i][j] / 255 for j in range(len(pred_vec[i]))], axis=0).squeeze())
# plt.show()
