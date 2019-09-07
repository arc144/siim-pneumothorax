import pandas as pd
import numpy as np
import os
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
from data import DicomDataset, load_mask_counts, load_gt

sub_fn = 'x_ensemble/CLF_ADJUST_Both_ENS_0024.csv'
out_fn = os.path.join(os.path.dirname(sub_fn), 'ADJUST_V2_' + os.path.basename(
    sub_fn))
sub = load_gt(sub_fn, rle_key='EncodedPixels')
val = '_VAL' in sub_fn
if val:
  ids = load_mask_counts('train-rle.csv')
else:
  ids = load_mask_counts('sample_submission.csv')
adjusted_sub = {'ImageId': [], 'EncodedPixels': []}

num_removed = 0
num_added = 0
num_missed = 0
for image_id in tqdm(sub):
  rles = sub[image_id]
  num_masks = ids.get(image_id, 1)
  masks = DicomDataset.rles_to_mask(rles, merge_masks=False)
  num_pred = masks.max()
  if num_pred > num_masks:
    sizes = np.float32([(masks == i).sum() for i in range(1, num_pred + 1)])
    inds = np.argsort(-sizes)[:num_masks]
    inds = [range(1, num_pred + 1)[ind] for ind in inds]
    num_removed += len(sizes) - len(inds)
    rles = []
    for ind in inds:
      rles.append(DicomDataset.mask_to_rle((masks == ind).astype(
          'uint8') * 255))

  elif num_masks > num_pred and num_masks >= 2:
    if num_pred >= 1:
      sizes = np.float32([(masks == i).sum() for i in range(1, num_pred + 1)])
      inds = np.argsort(-sizes)[:num_masks]
      inds = [range(1, num_pred + 1)[ind] for ind in inds]
      num_removed += len(sizes) - len(inds)
      for kk in range(num_masks - num_pred):
        ind = inds[kk % len(inds)]
        rles.append(DicomDataset.mask_to_rle((masks == ind).astype(
            'uint8') * 255))
        num_added += 1
    else:
      num_missed += num_masks

  for rle in rles:
    adjusted_sub['EncodedPixels'].append(rle)
    adjusted_sub['ImageId'].append(image_id)

adjusted_sub = pd.DataFrame(adjusted_sub, columns=['ImageId', 'EncodedPixels'])
adjusted_sub.to_csv(out_fn, index=False)
print("Wrote to: %s" % out_fn)
print("Removed: %d | Added: %d | Missed: %d" % (num_removed, num_added,
    num_missed))
# assert len(sub) == adjusted_sub.ImageId.nunique()
print(len(sub))
