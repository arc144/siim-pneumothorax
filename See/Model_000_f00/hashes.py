from glob import glob
import pydicom
import os
import pandas as pd
import imagehash
import hashlib  # noqa
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from data import DicomDataset


plot = False
train_image_fns = sorted(glob(os.path.join(
    'dicom-images-train', '*/*/*.dcm')))

test_image_fns = sorted(glob(os.path.join(
    'dicom-images-test', '*/*/*.dcm')))
all_fns = train_image_fns + test_image_fns


def get_hash(fn):
  arr = pydicom.read_file(fn).pixel_array
  # h = hashlib.md5(Image.fromarray(arr).tobytes()).hexdigest()
  h = imagehash.phash(Image.fromarray(arr))
  return h


if not os.path.exists('current_ptx_hashes.p'):
  with ThreadPoolExecutor() as e:
    hh = list(tqdm(e.map(get_hash, all_fns),
        total=len(all_fns)))
  with open('current_ptx_hashes.p', 'wb') as f:
    pickle.dump(hh, f)
else:
  with open('current_ptx_hashes.p', 'rb') as f:
    hh = pickle.load(f)

with open('nih_ptx_hashes.p', 'rb') as f:
  nih = pickle.load(f)

ptx_hashes = {v: k for k, v in nih.items()}
num_found = 0
sub_fn = 'x_ensemble/CLF_ADJUST_Both_ENS_V3_0049.csv'
sub = pd.read_csv(sub_fn)
sub = {k: v for k, v in zip(sub['ImageId'], sub['EncodedPixels'])}
offset = len(train_image_fns)
num_match = 0
num_not_in_nih_ptx = 0
num_ptx = 0

out_fn = sub_fn.replace('_V3', '_V4')
nih_ids, nih_rles = [], []
for k, h in tqdm(enumerate(hh[offset:]), total=len(hh[offset:])):
  fn = all_fns[k + offset]
  img_id = DicomDataset.fn_to_id(fn)
  p = sub[img_id]
  nih_ids.append(img_id)
  if h in ptx_hashes:
    nih_rles.append(p)
    num_ptx += 1
    if '-1' not in p:
      num_match += 1
      print("%s -> %s" % (fn, ptx_hashes[h]))
      if plot:
        img = pydicom.read_file(fn).pixel_array
        plt.imshow(img, cmap='bone')
        plt.show()

  else:
    if '-1' not in p:
      num_not_in_nih_ptx += 1
      nih_rles.append('-1')
    else:
      nih_rles.append(p)

print("Predicted as ptx and not in nih: %d" % (num_not_in_nih_ptx))
print("Num ptx (nih): %d (290 are there)" % num_ptx)
print("Num NIH ptx: %d" % len(nih))
nih_sub = pd.DataFrame({'ImageId': nih_ids, 'EncodedPixels': nih_rles})
nih_sub.to_csv(out_fn, index=False, columns=['ImageId', 'EncodedPixels'])
print("Wrote to: %s" % out_fn)
