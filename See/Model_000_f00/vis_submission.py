import argparse
import pandas as pd
import pydicom
import os
import numpy as np
from glob import glob
import cv2
from collections import defaultdict
from matplotlib import pyplot as plt
from data import DicomDataset, load_gt


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', type=str)
  parser.add_argument('--show-empty', action='store_true')
  parser.add_argument('--seed', type=int, default=32)
  parser.add_argument('--height', type=int, default=1024)

  args = parser.parse_args()
  sub = pd.read_csv(args.fn)
  np.random.seed(args.seed)
  if ' EncodedPixels' in sub.columns:
    sub['EncodedPixels'] = sub[' EncodedPixels']
    sub = sub[['ImageId', 'EncodedPixels']]

  sub['EncodedPixels'] = sub['EncodedPixels'].apply(lambda x: x if x != ' -1'
      else '-1')
  gt = load_gt('train-rle.csv')
  train_fns = sorted(glob('dicom-images-train/*/*/*.dcm'))
  test_fns = sorted(glob('dicom-images-test/*/*/*.dcm'))
  all_fns = train_fns + test_fns
  id_to_fn = {DicomDataset.fn_to_id(fn): fn for fn in all_fns}
  sub_ = defaultdict(list)
  for iid, rle in zip(sub['ImageId'], sub['EncodedPixels']):
    sub_[iid].append(rle)
  sub = sub_
  num_mask = sum(1 for k, v in sub.items() if v[0] != '-1')
  num_one_mask = sum(1 for k, v in sub.items() if v[0] != '-1' and len(v) == 1)
  num_more_mask = sum(1 for k, v in sub.items() if v[0] != '-1' and len(v) >= 2)
  print("%d of %d have a mask" % (num_mask, len(sub)))
  print("%d have 1, %d 2 or more" % (num_one_mask, num_more_mask))
  img_ids = sorted(sub.keys())
  np.random.shuffle(img_ids)
  for img_id in img_ids:
    img_fn = id_to_fn[img_id]
    rles = sub[img_id]
    if not args.show_empty:
      if rles[0] == '-1':
        continue
    print("%d masks" % len(rles))
    dcm = pydicom.dcmread(img_fn)
    view = dcm.ViewPosition
    print(view)
    img = dcm.pixel_array
    mask = DicomDataset.rles_to_mask(rles, merge_masks=False)
    if args.height != 1024:
      img = cv2.resize(img, (args.height, args.height),
          interpolation=cv2.INTER_NEAREST)
      mask = cv2.resize(mask, (args.height, args.height),
          interpolation=cv2.INTER_NEAREST)

    gt_mask = None
    if img_id in gt:
      gt_rles = gt[img_id]
      gt_mask = DicomDataset.rles_to_mask(gt_rles, merge_masks=False)
      gt_mask = cv2.resize(gt_mask, (args.height, args.height),
          interpolation=cv2.INTER_NEAREST)

      if gt_mask.max() == 0:
        continue
    # for j in range(0, 512, 16):
    #   img[:, j] = 255
    #   img[j, :] = 255
    #   mask[:, j] = mask.max()
    #   mask[j, :] = mask.max()

    nc = 2 if gt_mask is None else 3
    plt.subplot(1, nc, 1)
    plt.title(os.path.splitext(img_id)[-1])
    plt.imshow(img, cmap='bone')
    plt.axis('off')
    plt.subplot(1, nc, 2)
    plt.title('PRED: ' + str(mask.max()))
    plt.imshow(mask, cmap='bone', alpha=0.4)
    plt.axis('off')
    if gt_mask is not None:
      vis = np.dstack([img.copy()] * 3)
      vis[gt_mask > 0] = (0, 255, 0)
      vis[mask > 0] = 0.3 * vis[mask > 0] + 0.7 * np.float32([255, 0, 0])
      plt.subplot(1, nc, 3)
      plt.title('GT: ' + str(gt_mask.max()))
      plt.imshow(vis, cmap='bone')

      plt.axis('off')
    plt.show()


if __name__ == '__main__':
  main()
