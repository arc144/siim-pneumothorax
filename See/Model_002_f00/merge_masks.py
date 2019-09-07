import cv2  # noqa
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
from data import DicomDataset

sub_fn = 'Both_SEG_logdir_037_f04/Both_SEG_logdir_037_f04/Both_SEG_sub_37_f04_VAL.csv'  # noqa
out_fn = os.path.join(os.path.dirname(sub_fn),
    'MERGE_V3_' + os.path.basename(sub_fn))
sub = pd.read_csv(sub_fn)

sub_ = defaultdict(list)
for img_id, rle in zip(sub['ImageId'], sub['EncodedPixels']):
  sub_[img_id].append(rle)

sub = sub_
image_ids, sub_rles = [], []
for img_id, rles in tqdm(sub.items()):
  mask = np.zeros((1024, 1024), dtype='uint8')
  for rle in rles:
    if '-1' in rle:
      continue
    m = DicomDataset.rles_to_mask([rle])
    mask[m == 1] = 1
  image_ids.append(img_id)
  sub_rles.append(DicomDataset.mask_to_rle(mask * 255))

sub = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': sub_rles})
sub.to_csv(out_fn, index=False, columns=['ImageId', 'EncodedPixels'])
print("Wrote to: %s" % out_fn)
