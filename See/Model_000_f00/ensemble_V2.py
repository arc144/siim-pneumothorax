from zipfile import ZipFile
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm

from data import DicomDataset, load_mask_counts


ensemble_id = 24
p_clf = 0.65
p_seg = 0.1
clf_adjust = True
os.makedirs('x_ensemble', exist_ok=True)
# fns = [
#   # 'Both_SEG_logdir_029_f01/f01-PREDS_VAL.zip',
#   'Both_SEG_logdir_033_f00/f00-PREDS_VAL.zip',
#   'Both_SEG_logdir_034_f00/f00-PREDS_VAL.zip',
# ]

# fns = [
#   'Both_SEG_logdir_013_f00/f00-PREDS.zip',
#   'Both_SEG_logdir_014_f00/f00-PREDS.zip',
#   'Both_SEG_logdir_015_f00/f00-PREDS.zip',
#   'Both_SEG_logdir_013_f01/f01-PREDS.zip',
#   'Both_SEG_logdir_014_f01/f01-PREDS.zip',
#   'Both_SEG_logdir_015_f01/f01-PREDS.zip',
#   'Both_SEG_logdir_013_f02/f02-PREDS.zip',
#   'Both_SEG_logdir_014_f02/f02-PREDS.zip',
#   'Both_SEG_logdir_015_f02/f02-PREDS.zip',
#   'Both_SEG_logdir_013_f03/f03-PREDS.zip',
#   'Both_SEG_logdir_014_f03/f03-PREDS.zip',
#   'Both_SEG_logdir_015_f03/f03-PREDS.zip',
#   'Both_SEG_logdir_013_f04/f04-PREDS.zip',
#   'Both_SEG_logdir_014_f04/f04-PREDS.zip',
#   'Both_SEG_logdir_015_f04/f04-PREDS.zip',
# ]

fns = [
  'Both_SEG_logdir_037_f00/Both_SEG_logdir_037_f00/f00-PREDS.zip',
  'Both_SEG_logdir_037_f01/Both_SEG_logdir_037_f01/f01-PREDS.zip',
  'Both_SEG_logdir_037_f02/Both_SEG_logdir_037_f02/f02-PREDS.zip',
  'Both_SEG_logdir_037_f03/Both_SEG_logdir_037_f03/f03-PREDS.zip',
  'Both_SEG_logdir_037_f04/Both_SEG_logdir_037_f04/f04-PREDS.zip',

  # 'Both_SEG_logdir_034_f00/f00-PREDS.zip',
  # 'Both_SEG_logdir_034_f01/f01-PREDS.zip',
  # 'Both_SEG_logdir_034_f02/f02-PREDS.zip',
  # 'Both_SEG_logdir_034_f03/f03-PREDS.zip',
  # 'Both_SEG_logdir_034_f04/f04-PREDS.zip',
]

val = '_VAL' in fns[0]
if val:
  ids = load_mask_counts('train-rle.csv')
else:
  ids = load_mask_counts('sample_submission.csv')


prefix = fns[0].split('_')[0]
out_fn = os.path.join('x_ensemble', '%s_ENS_%04d%s.csv' % (prefix, ensemble_id,
    '_VAL' if val else ''))

num_passed = 0
still_not_passed = 0
if clf_adjust:
  out_fn = os.path.join(os.path.dirname(out_fn), 'CLF_ADJUST_' +
      os.path.basename(out_fn))

handels = [ZipFile(fn) for fn in fns]


image_ids, rles = [], []
pngs = ZipFile(fns[0]).namelist()
for png in tqdm(pngs):
  image_id = os.path.splitext(png)[0]
  p_ensemble = 0.0
  for handle in handels:
    with handle.open(png) as f:
      img = cv2.imdecode(np.frombuffer(f.read(), 'uint8'), 0)
      p = np.float32(img) / 255
      p_ensemble += p / len(fns)

  num_masks = ids.get(image_id, 1)
  passed = False
  # classify
  clf_mask = (p_ensemble > p_clf)
  if clf_mask.sum() == 0:
    if num_masks >= 2 and clf_adjust:
      print("Passing classifier: %s" % image_id)
      num_passed += 1
      passed = True
    else:
      image_ids.append(image_id)
      rles.append('-1')
      continue

  # segment
  mask = (p_ensemble > p_seg).astype('uint8')
  mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
  if passed:
    current_p_seg = p_seg
    while mask.sum() == 0 and current_p_seg > 0:
      current_p_seg -= 0.1
      print("Lowering p_seg to: %.2f" % current_p_seg)
      mask = (p_ensemble > current_p_seg).astype('uint8')
    if current_p_seg <= 0:
      still_not_passed += 1
      print("Still not passed ... ")

  if mask.sum() == 0:
    image_ids.append(image_id)
    rles.append('-1')
    continue

  labels = cv2.connectedComponents(mask, connectivity=8)[1]
  num_pred_masks = labels.max()
  for ind in range(1, num_pred_masks + 1):
    m = (labels == ind).astype('uint8')
    rle = DicomDataset.mask_to_rle(m * 255)
    image_ids.append(image_id)
    rles.append(rle)

sub = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': rles})
sub.to_csv(out_fn, columns=['ImageId', 'EncodedPixels'], index=False)
print("Wrote to: %s" % out_fn)
if clf_adjust:
  print("%d (could) have passed classifier, %d did" % (
      num_passed, num_passed - still_not_passed))
