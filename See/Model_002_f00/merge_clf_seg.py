import pandas as pd

merge_id = 35
out_fn = 'CLFSEGMERGE_%04d.csv' % merge_id
clf_fn = 'x_ensemble/MERGE_V3_ADJUST_V2_CLF_ADJUST_Both_ENS_0022.csv'
seg_fn = 'x_ensemble/CLF_ADJUST_Both_ENS_V3_0045.csv'

clf = pd.read_csv(clf_fn)
clf = {k: v for k, v in zip(clf['ImageId'], clf['EncodedPixels'])}
seg = pd.read_csv(seg_fn)
seg = {k: v for k, v in zip(seg['ImageId'], seg['EncodedPixels'])}

num_replaced = 0
image_ids, rles = [], []
for k, v in clf.items():
  image_ids.append(k)
  if '-1' in v:
    rles.append(v)
    continue
  rles.append(seg[k])
  num_replaced += 1

print("Replaced %d of %d" % (num_replaced, len(clf)))
merged = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': rles})
merged.to_csv(out_fn, index=False)
print("Wrote to: %s" % out_fn)
