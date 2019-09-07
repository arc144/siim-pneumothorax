import torch as th
th.backends.cudnn.deterministic = True  # noqa
th.backends.cudnn.benchmark = False  # noqa
from collections import defaultdict  # noqa
from tqdm import tqdm
import torch.utils.data as td
import os
import cv2
from matplotlib import pyplot as plt  # noqa
import pandas as pd
from zipfile import ZipFile
import numpy as np  # noqa
from glob import glob

from segmentation_model import FPNSegmentation
from data import DicomDataset, load_gt


def seed_all():
  SEED = 32
  np.random.seed(SEED)
  th.manual_seed(SEED)
  th.cuda.manual_seed(SEED)


def create_submission(model, loader, config, pred_zip=None, tta=True):
  if os.path.exists(pred_zip):
    os.remove(pred_zip)

  sub = []
  model.eval()
  with th.no_grad():
    for ret in tqdm(loader):
      X, fns = ret[:2]
      X = X.to(config.device).float()
      # that squeeze index is important! (in case of X.shape[0] == 1)
      y_pred = model(X).squeeze(1).cpu().numpy()
      if tta:
        y_pred_flip = th.flip(model(th.flip(X, (-1, ))), (-1, )).squeeze(
            1).cpu().numpy()
        y_pred = 0.5 * (y_pred + y_pred_flip)

      for j in range(len(y_pred)):
        img_id = DicomDataset.fn_to_id(fns[j])
        yp = y_pred[j]
        if pred_zip is not None:
          pred_fn = img_id + '.png'
          yp_img = np.uint8(yp * 255)
          img_bytes = cv2.imencode('.png', yp_img)[1].tobytes()
          with ZipFile(pred_zip, 'a') as f:
            f.writestr(pred_fn, img_bytes)

        assert yp.shape == (1024, 1024), yp.shape
        # classify
        clf_mask = (yp >= config.p_clf).astype('uint8')
        if clf_mask.sum() == 0:
          sub.append((img_id, '-1'))
          continue

        # segment
        mask = (yp >= config.p_seg).astype('uint8')
        assert mask.shape == (1024, 1024), mask.shape
        if mask.sum() == 0:
          sub.append((img_id, '-1'))
          continue

        rle = DicomDataset.mask_to_rle(mask * 255)
        sub.append((img_id, rle))

  image_ids = [s[0] for s in sub]
  encoded_pixels = [s[1] for s in sub]
  sub = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
  return sub


def main(config):
  seed_all()
  train_image_fns = sorted(glob(os.path.join(config.train_dir, '*/*/*.dcm')))
  test_image_fns = sorted(glob(os.path.join(config.test_dir, '*/*/*.dcm')))

  # assert len(train_image_fns) == 10712
  # assert len(test_image_fns) == 1377

  gt = load_gt(config.train_rle)
  # create folds
  np.random.shuffle(train_image_fns)

  folds = np.arange(len(train_image_fns)) % config.num_folds
  val_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] == config.fold]
  train_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] != config.fold]
  # remove not-used files:
  # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98478#latest-572385  # noqa
  train_image_fns = [fn for fn in train_image_fns
      if DicomDataset.fn_to_id(fn) in gt]
  val_image_fns = [fn for fn in val_image_fns
      if DicomDataset.fn_to_id(fn) in gt]

  print("VAL: ", len(val_image_fns), val_image_fns[0])
  print("TRAIN: ", len(train_image_fns), train_image_fns[0])
  if config.submit_val:
    test_image_fns = val_image_fns

  test_ds = DicomDataset(test_image_fns, gt_rles=gt, height=config.height,
      width=config.height)
  test_ds.cache()
  test_loader = td.DataLoader(test_ds, batch_size=config.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=False, drop_last=False)

  model = FPNSegmentation(config.slug)
  print("Loading: %s" % config.weights)
  r = model.load_state_dict(th.load(config.weight))
  from IPython import embed; embed()
  model = model.to(config.device).float()
  # model = apex.amp.initialize(model, opt_level="O1")
  model.eval()
  sub = create_submission(model, test_loader, config,
      pred_zip=config.pred_zip, tta=False)
  sub.to_csv(config.submission_fn, index=False)
  print("Wrote to %s" % config.submission_fn)


class Config:
  def as_dict(self):
    return vars(self)
  def __str__(self):
    return str(self.as_dict())
  def __repr__(self):
    return str(self)


if __name__ == '__main__':
  config = Config()
  config.train_dir = 'dicom-images-train'
  config.test_dir = 'dicom-images-test'
  config.height = 640
  config.batch_size = 16
  config.fold = 0
  config.num_folds = 5
  config.device = 'cuda'
  config.p_clf = 0.6
  config.p_seg = 0.2
  config.tta = True
  config.train_rle = 'train-rle.csv'
  config.slug = 'r50d'
  config.weight = 'Both_SEG_logdir_104_f00/f00-ep-0016-val_dice-0.5837@0.20.pth'
  for submit_val in [True, False]:  # [True, False]
    config.submit_val = submit_val
    dn = os.path.dirname(config.weight)
    bn = os.path.basename(config.weight)
    config.submission_fn = os.path.join(dn, bn.split('-')[0] +
        '%s-PREDS%s.csv' % ('-TTA' if config.tta else '-V2',
        '_VAL' if config.submit_val else ''))
    print("Saving to: %s" % config.submission_fn)
    config.pred_zip = config.submission_fn.replace('.csv', '.zip')
    print(config)
    main(config)
