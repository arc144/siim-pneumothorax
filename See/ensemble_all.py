from zipfile import ZipFile
import pandas as pd
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt  # noqa
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
from data import DicomDataset


ensemble_id = 3
p_clf = 0.4995
reduction_fn = lambda x: (x.mean() + x.max()) / 2  #  np.mean  # np.mean  # noqa
p_seg = 0.2
os.makedirs('x_ensemble', exist_ok=True)
fns = [
  'Model_000_f00/f00-PREDS.zip',
  'Model_000_f01/f01-PREDS.zip',
  'Model_000_f02/f02-PREDS.zip',
  'Model_000_f03/f03-PREDS.zip',
  'Model_000_f04/f04-PREDS.zip',

  'Model_001_f00/f00-PREDS.zip',
  'Model_001_f01/f01-PREDS.zip',
  'Model_001_f02/f02-PREDS.zip',
  'Model_001_f03/f03-PREDS.zip',
  'Model_001_f04/f04-PREDS.zip',

  'Model_002_f00/f00-PREDS.zip',
  'Model_002_f01/f01-PREDS.zip',
  'Model_002_f02/f02-PREDS.zip',
  'Model_002_f03/f03-PREDS.zip',
  'Model_002_f04/f04-PREDS.zip',
  'Model_002_f05/f05-PREDS.zip',
  'Model_002_f06/f06-PREDS.zip',
]

val = '_VAL' in fns[0]
prefix = fns[0].split('_')[0]
out_fn = os.path.join('x_ensemble', '%s_ENS_V5_%04d%s.csv' % (
    prefix, ensemble_id, '_VAL' if val else ''))

handels = [ZipFile(fn) for fn in fns]
image_ids, rles = [], []
pngs = ZipFile(fns[0]).namelist()
num_clf_empty = 0
for png in tqdm(pngs):
  image_id = os.path.splitext(png)[0]
  p_ensemble = 0.0
  for handle in handels:
    with handle.open(png) as f:
      img = cv2.imdecode(np.frombuffer(f.read(), 'uint8'), 0)
      p = np.float32(img) / 255
      # plt.imshow(p); plt.show()
      p_ensemble += p / len(fns)

  # classify
  p_reduced = reduction_fn(p_ensemble)
  if p_reduced < p_clf:
    image_ids.append(image_id)
    rles.append('-1')
    num_clf_empty += 1
    continue

  # segment
  mask = (p_ensemble > p_seg).astype('uint8')
  if mask.sum() == 0:
    image_ids.append(image_id)
    rles.append('-1')
    continue

  image_ids.append(image_id)
  rles.append(DicomDataset.mask_to_rle(mask * 255))

sub = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': rles})
sub.to_csv(out_fn, columns=['ImageId', 'EncodedPixels'], index=False)
print("Wrote to: %s" % out_fn)
print("Empty (clf): ", num_clf_empty)
