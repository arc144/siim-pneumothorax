import numpy as np
import pydicom
import cv2
from glob import glob
import torch.utils.data as data
import os
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
import pandas as pd
from collections import defaultdict, Counter  # noqa
import torch as th

from albumentations import Compose, ShiftScaleRotate  # noqa
# datetime.utcfromtimestamp(float('.'.join(a.iloc[0, 0].split('.')[-2:]))).
# strftime(('%Y-%m-%d %H:%M:%S'))


class DicomDataset(data.Dataset):
  def __init__(self, image_fns, gt_rles=None,
               augment=False,
               train_dicom_dir='dicom-images-train',
               test_dicom_dir='dicom-images-test'):
    self.image_fns = image_fns
    self.gt_rles = gt_rles
    self.augment = augment
    self.aug = Compose([
      ShiftScaleRotate(p=0.9, rotate_limit=10,
          border_mode=cv2.BORDER_CONSTANT)])
    self.encoded_cache = None

  def cache(self):
    self.encoded_cache = {}
    print("Caching ... ")
    with ThreadPoolExecutor() as e:
      encoded_imgs = list(tqdm(e.map(self.read_encoded, self.image_fns),
          total=len(self.image_fns)))
    for fn, encoded_channels in zip(self.image_fns, encoded_imgs):
      self.encoded_cache[fn] = encoded_channels

  def read_encoded(self, fn):
    pixels = self.read_pixels(fn)
    encoded_pixels = cv2.imencode('.png', pixels)[1]
    return encoded_pixels

  def read_pixels(self, fn):
    return pydicom.read_file(fn).pixel_array

  def decode_image(self, encoded_img):
    img = cv2.imdecode(encoded_img, 0)
    return img

  @staticmethod
  def fn_to_id(fn):
    return os.path.splitext(os.path.basename(fn))[0]

  @staticmethod
  def rles_to_mask(rles, height=1024, width=1024, merge_masks=True):
    mask = np.zeros(width * height, dtype=np.uint8)
    pos_value = 1
    assert isinstance(rles, list)
    for rle in rles:
      if rle == '-1' or rle == ' -1':
        continue
      array = np.asarray([int(x) for x in rle.split()])
      starts = array[0::2]
      lengths = array[1::2]
      current_position = 0
      for index, start in enumerate(starts):
          current_position += start
          mask[current_position:current_position + lengths[index]] = pos_value
          current_position += lengths[index]
      if not merge_masks:
        pos_value += 1
    return mask.reshape(width, height).T

  @staticmethod
  def mask_to_rle(img, mask_value=255, transpose=True):
    img = np.int32(img)
    if transpose:
      img = img.T
    img = img.flatten()
    img[img == mask_value] = 1
    pimg = np.pad(img, 1, mode='constant')
    diff = np.diff(pimg)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    rle = []
    previous_end = 0
    for start, end in zip(starts, ends):
      relative_start = start - previous_end
      length = end - start
      previous_end = end
      rle.append(str(relative_start))
      rle.append(str(length))
    if len(rle) == 0:
      return "-1"
    return " ".join(rle)

  def __getitem__(self, index, to_tensor=True):
    fn = self.image_fns[index]
    contrast = False
    flip = False
    if self.augment:
      contrast = th.FloatTensor(1).uniform_(0.0, 1.0).item() < 0.5
      flip = th.FloatTensor(1).uniform_(0.0, 1.0).item() < 0.5

    if self.encoded_cache is not None:
      img = self.decode_image(self.encoded_cache[fn])
    else:
      img = self.read_pixels(fn)

    if contrast:
      alpha = th.FloatTensor(1).uniform_(0.75, 1.25).item()
      img = (alpha * (np.float32(img) - 127.5)) + 127.5
      img = np.uint8(np.clip(img, 0, 255))

    if self.gt_rles is not None:
      gt_rles = self.gt_rles[self.fn_to_id(fn)]
      mask = self.rles_to_mask(gt_rles)
      if self.augment:
        augmented = self.aug(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask']
        if flip:
          img = img[:, ::-1].copy()
          mask = mask[:, ::-1].copy()

      if to_tensor:
        if img.ndim == 2:
          img = np.expand_dims(img, 0)
        elif img.ndim == 3:
          img = img.transpose((2, 0, 1))
        else:
          assert False, img.ndim

        img = th.from_numpy(img.copy())
        mask = np.expand_dims(mask, 0)
        mask = th.from_numpy(mask.copy())

      return img, fn, mask

    assert not self.augment, "Don't"
    if to_tensor:
      if img.ndim == 2:
        img = np.expand_dims(img, 0)
      elif img.ndim == 3:
        img = img.transpose((2, 0, 1))
      else:
        assert False, img.ndim
      img = th.from_numpy(img.copy())
    return img, fn

  def __len__(self):
    return len(self.image_fns)


class MultiScaleDicomDataset(data.Dataset):
  def __init__(self, single_scale_ds, scales):
    super().__init__()
    self.scales = scales
    self.multi_scale_ds = []
    for scale in scales:
      scale_ds = deepcopy(single_scale_ds)
      scale_ds.height = single_scale_ds.height + scale
      scale_ds.width = single_scale_ds.width + scale
      self.multi_scale_ds.append(scale_ds)

  def __len__(self):
    return len(self.multi_scale_ds[0])

  def __getitem__(self, index):
    ret = []
    for ds in self.multi_scale_ds:
      ret.append(ds[index])
    return ret


def load_gt(fn, rle_key=' EncodedPixels'):
  rles = pd.read_csv(fn, dtype={'ImageId': str, rle_key: str})
  rles_ = defaultdict(list)
  for img_id, rle in zip(rles['ImageId'], rles[rle_key]):
    rles_[img_id].append(rle)
  return rles_


def load_mask_counts(fn):
  sample = pd.read_csv(fn)
  id_count = Counter(sample.ImageId)
  # pneumo_end = 0
  # for k, image_id in enumerate(sample.ImageId):
  #   if id_count[image_id] >= 2:
  #     pneumo_end = k
  # print("End: ", pneumo_end)
  dups = {image_id: c for image_id, c in id_count.items() if c >= 2}
  print("%d of %d leaked as CONTAINING MASK" % (len(dups), len(id_count)))
  return dups


def split_by_view(fns):
  pa_fns, ap_fns = [], []
  for fn in tqdm(fns):
    vp = pydicom.read_file(fn).ViewPosition
    if vp == 'PA':
      pa_fns.append(fn)
    elif vp == 'AP':
      ap_fns.append(fn)
    else:
      assert False
  return pa_fns, ap_fns


if __name__ == '__main__':
  from matplotlib import pyplot as plt
  np.random.seed(321)
  th.manual_seed(321)
  gt = load_gt('train-rle.csv')
  train_image_fns = sorted(glob(os.path.join('dicom-images-train',
      '*/*/*.dcm')))
  test_image_fns = sorted(glob(os.path.join('dicom-images-test',
      '*/*/*.dcm')))

  # remove empty masks from training data
  non_empty_gt = {k: v for k, v in gt.items() if v[0] != ' -1'}
  train_image_fns = [fn for fn in train_image_fns if
      DicomDataset.fn_to_id(fn) in non_empty_gt]
  # subset
  train_image_fns = train_image_fns[2:3]
  print("[Non-EMPTY] TRAIN: ", len(train_image_fns), os.path.basename(
      train_image_fns[0]))
  train_ds = DicomDataset(train_image_fns, gt_rles=gt, augment=True)
  train_ds.cache()

  # multi_scale_ds = MultiScaleDicomDataset(train_ds, [-64, 0, 64])
  for k in range(500):
    index = np.random.randint(len(train_ds))
    ret = train_ds.__getitem__(index)
    img, fn, mask = ret
    img, mask = img.squeeze(), mask.squeeze()
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.show()
    # for j, (img, fn, mask) in enumerate(ret):
    #   print(fn)
    #   img, mask = img.squeeze(), mask.squeeze()
    #   plt.subplot(3, 2, j * 2 + 1)
    #   plt.imshow(img, cmap='bone')
    #   plt.subplot(3, 2, j * 2 + 2)
    #   plt.imshow(mask)
    # plt.show()
