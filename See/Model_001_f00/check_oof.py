import pandas as pd
import pydicom
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from data import DicomDataset, load_gt
from matplotlib import pyplot as plt
import pickle


plot = False
oof_fns = [
  'Both_SEG_logdir_073_f00/Both_SEG_sub_73_f00_VAL.csv',
  'Both_SEG_logdir_073_f01/Both_SEG_sub_73_f01_VAL.csv',
  'Both_SEG_logdir_073_f02/Both_SEG_sub_73_f02_VAL.csv',
  'Both_SEG_logdir_073_f03/Both_SEG_sub_73_f03_VAL.csv',
  'Both_SEG_logdir_073_f04/Both_SEG_sub_73_f04_VAL.csv',
]

with open('nih_ptx_hashes.p', 'rb') as f:
  nih = pickle.load(f)
  ptx_hashes = set(nih.values())


with open('current_ptx_hashes.p', 'rb') as f:
  hh = pickle.load(f)

preds = [pd.read_csv(fn) for fn in oof_fns]
gts = load_gt('train-rle.csv')
np.random.seed(123)
pred = pd.concat(preds)
pred = {k: v for k, v in zip(pred['ImageId'], pred['EncodedPixels'])}
train_image_fns = sorted(glob(os.path.join('dicom-images-train',
    '*/*/*.dcm')))
np.random.shuffle(train_image_fns)
num_fp, num_fp_in_ptx, num_fp_not_in_ptx = 0, 0, 0
image_ids, rles = [], []
num_missing = 0
for ind, fn in tqdm(enumerate(train_image_fns), total=len(train_image_fns)):
  img_id = DicomDataset.fn_to_id(fn)
  try:
    p = pred[img_id]
  except:
    num_missing += 1
    print(img_id)
    continue
  gt = gts[img_id]
  image_ids.append(img_id)
  rles.append(p)
  if len(gt) == 0:
    continue
  if '-1' not in p and '-1' in gt[0]:
    num_fp += 1
    is_in = False
    if hh[ind] in ptx_hashes:
      num_fp_in_ptx += 1
      is_in = True
    else:
      num_fp_not_in_ptx += 1
    # if plot and is_in:
    if True:
      pred_mask = DicomDataset.rles_to_mask([p])
      print(pred_mask.sum())
      if pred_mask.sum() < 2000 and is_in:
        num_fp_in_ptx -= 1
        is_in = False
      if plot or is_in:
        gt_mask = DicomDataset.rles_to_mask(gt)
        img = pydicom.read_file(fn).pixel_array
        img = np.dstack([img] * 3)
        img[gt_mask > 0] = 0.7 * img[gt_mask > 0] + 0.3 * np.float32(
            [0, 255, 0])
        img[pred_mask > 0] = 0.7 * img[pred_mask > 0] + \
            0.3 * np.float32([255, 0, 0])
        plt.imshow(img, cmap='bone', alpha=0.8)
        plt.title(os.path.basename(fn))
        plt.axis('off')
        plt.show()


oof = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': rles})
print("FP: ", num_fp)
out_fn = 'oof_preds.csv'
oof.to_csv(out_fn, index=False, columns=['ImageId',
    'EncodedPixels'])
print("Wrote to: %s" % out_fn)
print("Missing: %d" % num_missing)
print("IN %d, OUT: %d" % (num_fp_in_ptx, num_fp_not_in_ptx))
