import pandas as pd
import cv2
from matplotlib import pyplot as plt  # noqa
from data import DicomDataset


fn = 'submission.csv'
splitted_fn = 'SPLIT_' + fn
sub = pd.read_csv(fn)
img_ids, rles = [], []

for img_id, rle in zip(sub['ImageId'], sub['EncodedPixels']):
  if '-1' in rle:
    img_ids.append(img_id)
    rles.append(rle)
    continue
  mask = DicomDataset.rles_to_mask([rle])
  _, labels = cv2.connectedComponents(mask.astype('uint8'), connectivity=8)
  num_pred_masks = labels.max()
  for label_index in range(1, num_pred_masks + 1):
    m = (labels == label_index)
    rle = DicomDataset.mask_to_rle(m * 255)
    img_ids.append(img_id)
    rles.append(rle)

splitted_sub = pd.DataFrame({'ImageId': img_ids, 'EncodedPixels': rles})
splitted_sub.to_csv(splitted_fn, index=False, columns=[
    'ImageId', 'EncodedPixels'])
print("Wrote to: %s" % splitted_fn)
