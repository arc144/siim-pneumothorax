import pandas as pd

merge_id = 20
thresh = 0.95
seg_fn = 'SEG_logdir_003_f00/SEG_sub_3.csv'
clf_fn = 'ADJUST_hybrid.csv'
segmentation = pd.read_csv(seg_fn)
val = '_VAL' in seg_fn
sub_fn = 'MERGE_%04d%s.csv' % (merge_id, '_VAL' if val else '')
num_total = segmentation.ImageId.nunique()
seg_copy = segmentation.copy()

classification = pd.read_csv(clf_fn)
clf_copy = classification.copy()
classification = classification[classification['EncodedPixels'] == '-1']
print("%d of %d have mask" % (num_total - classification.shape[0], num_total))
classification['EncodedPixels'] = '-1'
classification = classification[['ImageId', 'EncodedPixels']]
classification_ids = classification.ImageId
segmentation = segmentation[~segmentation.ImageId.isin(classification_ids)]

sub = pd.concat([classification, segmentation])
sub.to_csv(sub_fn, index=False)
print("Wrote to: %s" % sub_fn)
num_ids = sub.ImageId.nunique()
if not val:
  assert num_ids == 1377, num_ids
else:
  assert clf_copy.ImageId.nunique() == seg_copy.ImageId.nunique()
