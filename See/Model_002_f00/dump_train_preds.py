import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import cv2
from zipfile import ZipFile

fns = [
  'Both_SEG_logdir_089_f00/f00-PREDS_VAL.zip',
  'Both_SEG_logdir_089_f01/f01-PREDS_VAL.zip',
  'Both_SEG_logdir_089_f02/f02-PREDS_VAL.zip',
  'Both_SEG_logdir_089_f03/f03-PREDS_VAL.zip',
  'Both_SEG_logdir_089_f04/f04-PREDS_VAL.zip',
]


out_fn = 'TRAIN_PREDS.csv'
image_ids, max_probs = [], []
handels = [ZipFile(fn) for fn in fns]
num_clf_empty = 0
for handle in handels:
  for png in tqdm(handle.namelist()):
    image_id = os.path.splitext(png)[0]
    with handle.open(png) as f:
      img = cv2.imdecode(np.frombuffer(f.read(), 'uint8'), 0)
      p = np.float32(img) / 255

    image_ids.append(image_id)
    max_probs.append(p.max())


preds = pd.DataFrame({'ImageId': image_ids, 'MaxProb': max_probs})
preds.to_csv(out_fn, index=False, columns=['ImageId', 'MaxProb'])
print("Wrote to: %s" % out_fn)
