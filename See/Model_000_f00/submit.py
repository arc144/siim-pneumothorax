import pandas as pd
from zipfile import ZipFile
import torch as th
import cv2
import numpy as np
import os
from glob import glob
import pydicom
from matplotlib import pyplot as plt
from segmentation_model import FPNSegmentation


def main():
  train_image_fns = sorted(glob(os.path.join(
      'dicom-images-train', '*/*/*.dcm')))
  m = {os.path.basename(fn): fn for fn in train_image_fns}
  ref_file = 'Both_SEG_logdir_001_f02/f00-PREDS_VAL.zip'
  slug = 'r50d'
  weight = 'Both_SEG_logdir_001_f02/f02-ep-0012-val_dice-0.5897@0.10.pth'
  model = FPNSegmentation(slug)
  model.load_state_dict(th.load(weight))
  model = model.cuda()
  model.eval()

  with ZipFile(ref_file) as f:
    for fn in f.namelist()[::10]:
      path = m[fn.replace('.png', '.dcm')]
      img = pydicom.read_file(path).pixel_array
      pimg = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
      X = th.from_numpy(pimg).unsqueeze(0).unsqueeze(0)
      with th.no_grad():
        X = X.cuda().float()
        y_pred = model(X).cpu().numpy().squeeze()
        y_pred_flip = th.flip(model(th.flip(X, (-1, ))),
            (-1, )).cpu().numpy().squeeze()
        # y_pred = 0.5 * (y_pred_flip + y_pred)
        y_pred = (y_pred * 255).astype(np.uint8)
      with f.open(fn) as h:
        pred = cv2.imdecode(np.frombuffer(h.read(), 'uint8'), 0)

      diff = y_pred != pred
      print("DIFF: ", diff.sum())
      plt.subplot(2, 2, 1)
      plt.imshow(img)
      plt.subplot(2, 2, 2)
      plt.imshow(y_pred)
      plt.subplot(2, 2, 3)
      plt.imshow(pred)
      plt.subplot(2, 2, 4)
      plt.imshow(diff)
      plt.show()


if __name__ == '__main__':
  main()
