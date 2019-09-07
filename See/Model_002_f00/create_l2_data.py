import os
from glob import glob
from tqdm import tqdm
from zipfile import ZipFile

os.makedirs('l2-images-train', exist_ok=True)
os.makedirs('l2-images-test', exist_ok=True)

train_fn_glob = 'Both_SEG_logdir_075_f*/f*PREDS_VAL.zip'

fns = sorted(glob(train_fn_glob))
for fn in tqdm(fns):
  with ZipFile(fn) as f:
    for img_fn in f.namelist():
      f.extract(img_fn, 'l2-images-train')

# TODO: more folds?
test_fn_glob = 'Both_SEG_logdir_075_f00/f00*PREDS.zip'
fns = sorted(glob(test_fn_glob))
for fn in tqdm(fns):
  with ZipFile(fn) as f:
    for img_fn in f.namelist():
      f.extract(img_fn, 'l2-images-test')

print("Done!")
