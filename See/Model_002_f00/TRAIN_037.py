# This is a backup from my Kernel for training 37
import torch as th
import torch.utils.data as data
from tqdm import tqdm_notebook as tqdm
import os
import time
import apex
from matplotlib import pyplot as plt  # noqa
import numpy as np
import pickle
import cv2
from glob import glob

"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific
   heads on top of it.

See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object
  Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified
  Architecture for Instance and Semantic Segmentation

"""

import torch
import torch.nn as nn

from torchvision import models


class FPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral
       connections. Can be used as feature extractor for object detection
       or segmentation.
    """

    def __init__(self, slug, num_filters=256, pretrained=True,
            num_input_channels=3):
        """Creates an `FPN` instance for feature extraction.

        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number fo input channels
        """

        super().__init__()
        assert pretrained

        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        else:
            assert False, "Bad slug: %s" % slug

        # adjust input channels
        if num_input_channels == 1:
            extracted_weight = self.resnet.conv1.weight[:, 0:1, :, :]
            self.resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3,
                bias=False)
            self.resnet.conv1.weight.data = extracted_weight.data
        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.lateral4 = Conv1x1(num_bottleneck_filters, num_filters)
        self.lateral3 = Conv1x1(num_bottleneck_filters // 2, num_filters)
        self.lateral2 = Conv1x1(num_bottleneck_filters // 4, num_filters)
        self.lateral1 = Conv1x1(num_bottleneck_filters // 8, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.interpolate(map4, scale_factor=2,
            mode="nearest")
        map2 = lateral2 + nn.functional.interpolate(map3, scale_factor=2,
            mode="nearest")
        map1 = lateral1 + nn.functional.interpolate(map2, scale_factor=2,
            mode="nearest")

        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4


class FPNSegmentation(nn.Module):
    """Semantic segmentation model on top of a Feature Pyramid Network (FPN).
    """

    def __init__(self, slug, num_classes=1, num_filters=128,
            num_filters_fpn=256, pretrained=True, num_input_channels=1,
            output_size=(1024, 1024)):
        """Creates an `FPNSegmentation` instance for feature extraction.

        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_classes: number of classes to predict
          num_filters: the number of filters in each segmentation head pyramid
                       level
          num_filters_fpn: the number of filters in each FPN output pyramid
                           level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number of input channels e.g. 3 for RGB
          output_size. Tuple[int, int] height, width
        """

        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(slug=slug, num_filters=num_filters_fpn,
            pretrained=pretrained, num_input_channels=num_input_channels)

        # The segmentation heads on top of the FPN

        self.head1 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head2 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head3 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head4 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))

        self.final = nn.Conv2d(4 * num_filters, num_classes, kernel_size=3,
            padding=1)
        self.sigmoid = nn.Sigmoid()
        self.output_size = output_size

    def forward(self, x):
        map1, map2, map3, map4 = self.fpn(x)
        map4 = nn.functional.interpolate(self.head4(map4), scale_factor=8,
            mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4,
            mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=2,
            mode="nearest")
        map1 = self.head1(map1)

        final = self.final(torch.cat([map4, map3, map2, map1], dim=1))

        return self.sigmoid(nn.functional.interpolate(final,
            size=self.output_size, mode="bilinear", align_corners=False))


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        return self.block(x)


import numpy as np
import pydicom
import cv2
import hashlib
import torch.utils.data as data
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm_notebook as tqdm
import pandas as pd
from collections import defaultdict
import torch as th

from albumentations import Compose, ShiftScaleRotate  # noqa
# datetime.utcfromtimestamp(float('.'.join(a.iloc[0, 0].split('.')[-2:]))).
# strftime(('%Y-%m-%d %H:%M:%S'))


class DicomDataset(data.Dataset):
  def __init__(self, image_fns, gt_rles=None, height=1024, width=1024,
               to_ram=False, preprocess_fn=lambda x: x / 127.5 - 1.0,
               augment=False, write_cache=True):
    self.image_fns = image_fns
    self.gt_rles = gt_rles
    self.height = height
    self.width = width
    self.to_ram = to_ram
    self.preprocess_fn = preprocess_fn
    self.augment = augment
    self.aug = Compose([
      ShiftScaleRotate(p=0.9, rotate_limit=10,
          border_mode=cv2.BORDER_CONSTANT)])
    self.write_cache = write_cache

    if self.to_ram:
      os.makedirs('cache', exist_ok=True)
      image_h = hashlib.md5((str(height) + str(width) + ','.join(
          self.image_fns)).encode()).hexdigest()
      cache_fn = os.path.join('cache', image_h + '.p')
      if os.path.exists(cache_fn):
        with open(cache_fn, 'rb') as f:
          self.imgs = pickle.load(f)
      else:
        print("Loading to RAM")
        with ThreadPoolExecutor() as e:
          self.imgs = list(tqdm(e.map(lambda fn: cv2.resize(pydicom.read_file(
              fn).pixel_array, (self.width, self.height),
              interpolation=cv2.INTER_LINEAR), self.image_fns),
              total=len(self.image_fns)))
        if self.write_cache:
          with open(cache_fn, 'wb') as f:
            pickle.dump(self.imgs, f)

  @staticmethod
  def fn_to_id(fn):
    return os.path.splitext(os.path.basename(fn))[0]

  @staticmethod
  def rles_to_mask(rles, height=1024, width=1024, merge_masks=True):
    mask = np.zeros(width * height, dtype=np.int32)
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

  def __getitem__(self, index, normalize=True, to_tensor=True):
    fn = self.image_fns[index]
    contrast = False
    flip = False
    if self.augment:
      contrast = np.random.rand() < 0.5
      flip = np.random.rand() < 0.5

    if self.to_ram:
      img = self.imgs[index]
    else:
      img = pydicom.read_file(fn).pixel_array
      img = cv2.resize(img, (self.width, self.height),
          interpolation=cv2.INTER_CUBIC)

    assert img.ndim == 2

    if contrast:
      alpha = np.random.uniform(0.8, 1.2)
      img = (alpha * (np.float32(img) - 127.5)) + 127.5
      img = np.uint8(np.clip(img, 0, 255))

    if self.gt_rles is not None:
      gt_rles = self.gt_rles[self.fn_to_id(fn)]
      mask = self.rles_to_mask(gt_rles)
      # mask = cv2.resize(mask, (self.height, self.height),
      #     interpolation=cv2.INTER_NEAREST)
      if self.augment:
        augmented = self.aug(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask']
        if flip:
          img = img[:, ::-1].copy()
          mask = mask[:, ::-1].copy()

      if normalize:
        img = np.float32(img)
        img = self.preprocess_fn(img)

      if to_tensor:
        img = np.expand_dims(img, 0)
        img = th.tensor(img, dtype=th.float32)
        mask = np.expand_dims(mask, 0)
        mask = th.tensor(mask, dtype=th.float32)

      return img, mask

    assert not self.augment, "Don't"
    if normalize:
      img = np.float32(img)
      img = self.preprocess_fn(img)

    if to_tensor:
      img = np.expand_dims(img, 0)
      img = th.tensor(img, dtype=th.float32)
    return img

  def __len__(self):
    return len(self.image_fns)


def load_gt(fn, rle_key=' EncodedPixels'):
  rles = pd.read_csv(fn, dtype={'ImageId': str, rle_key: str})
  rles_ = defaultdict(list)
  for img_id, rle in zip(rles['ImageId'], rles[rle_key]):
    rles_[img_id].append(rle)
  return rles_


def load_mask_counts(fn):
  sample = pd.read_csv(fn)
  id_count = defaultdict(int)
  for image_id in sample.ImageId:
    id_count[image_id] += 1
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


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class _LRSchedule(ABC):
    """ Parent of all LRSchedules here. """
    warn_t_total = False        # is set to True for schedules where progressing beyond t_total steps doesn't make sense
    def __init__(self, warmup=0.002, t_total=-1, **kw):
        """
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps (updates) are planned
        :param kw:
        """
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            logger.warning("t_total value of {} results in schedule not being applied".format(t_total))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        warmup = max(warmup, 0.)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.t_total < 0:
            return 1.
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            logger.warning(
                "Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly."
                    .format(ret, self.__class__.__name__))
            self.warned_for_t_total_at_progress = progress
        # end warning
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return 1.


class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Decreases learning rate from 1. to 0. over remaining `1 - warmup` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    warn_t_total = True
    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kw):
        """
        :param warmup:      see LRSchedule
        :param t_total:     see LRSchedule
        :param cycles:      number of cycles. Default: 0.5, corresponding to cosine decay from 1. at progress==warmup and 0 at progress==1.
        :param kw:
        """
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            return 0.5 * (1. + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)
        assert(cycles >= 1.)

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    """
    All training progress is divided in `cycles` (default=1.) parts of equal length.
    Every part follows a schedule with the first `warmup` fraction of the training steps linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        assert(warmup * cycles < 1.)
        warmup = warmup * cycles if warmup >= 0 else warmup
        super(WarmupCosineWithWarmupRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.
    """
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """
    warn_t_total = True
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)


SCHEDULES = {
    None:       ConstantLR,
    "none":     ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule
}
import pandas as pd  # noqa
import numpy as np
import argparse
from tqdm import tqdm_notebook as tqdm
from concurrent.futures import ThreadPoolExecutor


def score(yt, yp):
  assert yt.dtype == 'int32', yt.dtype
  assert yp.dtype == 'int32', yp.dtype
  assert yt.shape == (1024, 1024), yt.shape
  assert yt.shape == yp.shape, yp.shape
  num_gt_masks = yt.max()
  num_pred_masks = yp.max()
  if num_gt_masks == 0:
    if num_pred_masks != 0:
      score = (0, 'empty', 'non-empty')
    else:
      score = (1, 'empty', 'empty')
    return score
  per_image_scores = []
  matched_pred_indices = []
  for gt_index in range(1, num_gt_masks + 1):
    gt_mask = yt == gt_index
    best_dice_coeff = 0.
    best_pred_index = None
    for pred_index in range(1, num_pred_masks + 1):
      if pred_index in matched_pred_indices:
        continue
      pred_mask = yp == pred_index
      intersection = np.logical_and(gt_mask, pred_mask).sum()
      dice_coeff = (2 * intersection) / (gt_mask.sum() + pred_mask.sum())
      if dice_coeff > best_dice_coeff:
        best_dice_coeff = dice_coeff
        best_pred_index = pred_index

    matched_pred_indices.append(best_pred_index)
    per_image_scores.append(best_dice_coeff)

  # too many predictions
  per_image_scores.extend([0] * (num_pred_masks - len(matched_pred_indices)))
  score = (np.mean(per_image_scores), 'non-empty',
      'empty' if num_gt_masks == 0 else 'non-empty')
  return score


def run_server(prediction_fn, gt_fn):
  submission = load_gt(prediction_fn, rle_key='EncodedPixels')
  gt = load_gt(gt_fn)

  def compute_score(key):
    yt = DicomDataset.rles_to_mask(gt[key], merge_masks=False)
    yp = DicomDataset.rles_to_mask(submission[key], merge_masks=False)
    return score(yt, yp)

  scores = []
  keys = list(submission)

  with ThreadPoolExecutor(1) as e:
    scores = list(tqdm(e.map(compute_score, keys), total=len(keys)))

  empty_score = np.sum([s[0] for s in scores if s[1] == 'empty'])
  num_empty = sum(1 for s in scores if s[1] == 'empty')
  num_empty_pred = sum(1 for s in scores if s[-1] == 'empty')
  num_non_empty_pred = sum(1 for s in scores if s[-1] == 'non-empty')
  non_empty_score = np.sum([s[0] for s in scores if s[1] == 'non-empty'])
  num_non_empty = len(scores) - num_empty
  final_score = np.sum([s[0] for s in scores]) / len(scores)

  print("[GT: %5d | P: %5d] %012s %.4f | %.4f" % (num_empty, num_empty_pred,
      'Empty: ', empty_score / num_empty, empty_score / len(scores)))
  print("[GT: %5d | P: %5d] %012s %.4f | %.4f" % (num_non_empty,
      num_non_empty_pred, 'Non-Empty: ', non_empty_score / num_non_empty,
      non_empty_score / len(scores)))
  print("[%5d] Final: %.4f" % (len(scores), final_score))
  return final_score


import torch as th
from collections import defaultdict  # noqa
from tqdm import tqdm_notebook as tqdm
import torch.utils.data as data
import os
from matplotlib import pyplot as plt  # noqa
import pandas as pd
from zipfile import ZipFile
import numpy as np  # noqa
from glob import glob
import cv2


SEED = 32
FILL_MASK_RLE = '1 20'
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def create_submission(model, loader, fns, config, sample_ids=None,
      pred_zip=None, flip_tta=True):
  sub = []
  fn_index = 0
  model.eval()
  with th.no_grad():
    for X in tqdm(loader):
      if isinstance(X, list):
        X = X[0]
      X = X.to(config.device)
      # that squeeze index is important! (only one left!)
      y_pred = model(X).squeeze(1).cpu().numpy()
      if flip_tta:
        y_pred_flip = th.flip(model(th.flip(X, (-1, ))), (-1, )).squeeze(
            1).cpu().numpy()
        y_pred = 0.5 * (y_pred_flip + y_pred)

      for j in range(len(y_pred)):
        img_id = DicomDataset.fn_to_id(fns[fn_index])
        fn_index += 1
        yp = y_pred[j]
        if pred_zip is not None:
          pred_fn = img_id + '.png'
          yp_img = np.uint8(yp * 255)
          img_bytes = cv2.imencode('.png', yp_img)[1].tobytes()
          with ZipFile(pred_zip, 'a') as f:
            f.writestr(pred_fn, img_bytes)

        assert yp.shape == (1024, 1024), yp.shape
        # classify
        clf_mask = (yp >= config.p_clf).astype('uint8')
        if clf_mask.sum() == 0:
          sub.append((img_id, '-1'))
          continue

        # segment
        mask = (yp >= config.p_seg).astype('uint8')
        assert mask.shape == (1024, 1024), mask.shape
        if mask.sum() == 0:
          sub.append((img_id, '-1'))
          continue

        _, labels = cv2.connectedComponents(mask, connectivity=8)
        num_pred_masks = labels.max()
        for ind in range(1, num_pred_masks + 1):
          m = (labels == ind).astype('uint8')
          rle = DicomDataset.mask_to_rle(m * 255)
          sub.append((img_id, rle))

  image_ids = [s[0] for s in sub]
  encoded_pixels = [s[1] for s in sub]
  sub = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
  return sub




SEED = 32
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def bce(y_true, y_pred, weights=None):
  assert y_true.shape == y_pred.shape
  y_pred = th.clamp(y_pred, 1e-6, 1. - 1e-6)
  loss = -1.0 * y_true * th.log(y_pred) - (1. - y_true) * th.log(1. - y_pred)
  loss = th.mean(loss)
  return loss


def main(config):
  os.makedirs('cache', exist_ok=True)
  os.makedirs(config.logdir, exist_ok=True)
  print("Logging to: %s" % config.logdir)
  if not os.path.exists(config.train_dir):
    print("KERNEL ENV")
    config.train_dir = '../input/siim-train-test/siim/dicom-images-train'
    config.test_dir = '../input/siim-train-test/siim/dicom-images-test'
    config.sample_submission = '../input/siim-acr-pneumothorax-segmentation/' \
        'sample_submission.csv'
    config.train_rle = '../input/siim-train-test/siim/train-rle.csv'

  train_image_fns = sorted(glob(os.path.join(config.train_dir, '*/*/*.dcm')))
  test_image_fns = sorted(glob(os.path.join(config.test_dir, '*/*/*.dcm')))

  cache_fn = 'cache/view.p'
  if not os.path.exists(cache_fn):
    train_image_fns_PA, train_image_fns_AP = split_by_view(train_image_fns)
    test_image_fns_PA, test_image_fns_AP = split_by_view(test_image_fns)
    if not config.is_kernel:
      with open(cache_fn, 'wb') as f:
        pickle.dump([train_image_fns_PA, train_image_fns_AP,
            test_image_fns_PA, test_image_fns_AP], f)
  else:
    with open(cache_fn, 'rb') as f:
      train_image_fns_PA, train_image_fns_AP, test_image_fns_PA, \
          test_image_fns_AP = pickle.load(f)

  # assert len(train_image_fns_PA + train_image_fns_AP) == 10712
  # assert len(test_image_fns_PA + test_image_fns_AP) == 1377

  if config.view_position == 'PA':
    train_image_fns = train_image_fns_PA
    test_image_fns = test_image_fns_PA

  elif config.view_position == 'AP':
    train_image_fns = train_image_fns_AP
    test_image_fns = test_image_fns_AP

  elif config.view_position == 'Both':
    pass

  gt = load_gt(config.train_rle)
  # create folds
  np.random.shuffle(train_image_fns)
  folds = np.arange(len(train_image_fns)) % config.num_folds
  val_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] == config.fold]
  train_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] != config.fold]

  if config.drop_empty:
    # remove empty masks from training data
    non_empty_gt = {k: v for k, v in gt.items() if v[0] != ' -1'}
    train_image_fns = [fn for fn in train_image_fns if
        DicomDataset.fn_to_id(fn) in non_empty_gt]
    print("[Non-EMPTY] TRAIN: ", len(train_image_fns), os.path.basename(
        train_image_fns[0]))

  print("VAL: ", len(val_image_fns), os.path.basename(val_image_fns[0]))
  print("TRAIN: ", len(train_image_fns), os.path.basename(train_image_fns[0]))

  train_ds = DicomDataset(train_image_fns, gt_rles=gt, height=config.height,
      width=config.height, to_ram=True, augment=True,
      write_cache=not config.is_kernel)
  val_ds = DicomDataset(val_image_fns, gt_rles=gt, height=config.height,
      width=config.height, to_ram=True,
      write_cache=not config.is_kernel)

  val_loader = data.DataLoader(val_ds, batch_size=config.batch_size,
                               shuffle=False, num_workers=config.num_workers,
                               pin_memory=config.pin, drop_last=False)

  model = FPNSegmentation(config.slug)
  if config.weight is not None:
    model.load_state_dict(th.load(config.weight))
  model = model.to(config.device)

  optimizer = th.optim.Adam(model.parameters(), lr=config.lr,
      weight_decay=config.weight_decay)

  if config.apex:
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1",
                                           verbosity=0)

  updates_per_epoch = len(train_ds) // config.batch_size
  num_updates = int(config.epochs * updates_per_epoch)
  scheduler = WarmupLinearSchedule(warmup=config.warmup, t_total=num_updates)

  # training loop
  smooth = 0.1
  best_dice = 0.0
  best_fn = None
  global_step = 0
  for epoch in range(config.epochs):
    smooth_loss = None
    smooth_accuracy = None
    model.train()
    train_loader = data.DataLoader(train_ds, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.num_workers,
                                   pin_memory=config.pin, drop_last=True)
    progress = tqdm(total=len(train_ds), smoothing=0.01)
    for i, (X, y_true) in enumerate(train_loader):
      X = X.to(config.device)
      y_true = y_true.to(config.device)
      y_pred = model(X)
      loss = bce(y_true, y_pred, weights=None)
      if config.apex:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      lr_this_step = None
      if (i + 1) % config.accumulation_step == 0:
        optimizer.step()
        optimizer.zero_grad()
        lr_this_step = config.lr * scheduler.get_lr(global_step, config.warmup)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_this_step
        global_step += 1

      smooth_loss = loss.item() if smooth_loss is None else \
          smooth * loss.item() + (1. - smooth) * smooth_loss
      # print((y_true >= 0.5).sum().item())
      accuracy = th.mean(((y_pred >= 0.5) == (y_true >= 0.5)).to(
          th.float)).item()
      smooth_accuracy = accuracy if smooth_accuracy is None else \
          smooth * accuracy + (1. - smooth) * smooth_accuracy
      progress.set_postfix(loss='%.4f' % smooth_loss, accuracy='%.4f' %
            (smooth_accuracy), lr='%.5f' % (config.lr if lr_this_step is None
              else lr_this_step))
      progress.update(len(X))

    # validation loop
    model.eval()
    thresholds = np.arange(0.1, 0.7, 0.1)
    dice_coeffs = [[] for _ in range(len(thresholds))]
    progress = tqdm(enumerate(val_loader), total=len(val_loader))
    with th.no_grad():
      for i, (X, y_trues) in progress:
        X = X.to(config.device)
        y_trues = y_trues.to(config.device)
        y_preds = model(X)
        for yt, yp in zip(y_trues, y_preds):
          yt = (yt.squeeze().cpu().numpy() >= 0.5).astype('uint8')
          yt = cv2.connectedComponents(yt, connectivity=8)[1]
          yp = yp.squeeze().cpu().numpy()
          for dind, threshold in enumerate(thresholds):
            yp_ = (yp >= threshold).astype(np.uint8)
            yp_ = cv2.connectedComponents(yp_, connectivity=8)[1]
            sc = score(yt, yp_)
            dice_coeffs[dind].append(sc)

    best_threshold_ind = -1
    dice_coeff = -1
    for dind, threshold in enumerate(thresholds):
      dc = np.mean([x[0] for x in dice_coeffs[dind] if x[1] == 'non-empty'])
      # progress.write("Dice @%.2f: %.4f" % (threshold, dc))
      if dc > dice_coeff:
        dice_coeff = dc
        best_threshold_ind = dind

    dice_coeffs = dice_coeffs[best_threshold_ind]
    num_empty = sum(1 for x in dice_coeffs if x[1] == 'empty')
    num_total = len(dice_coeffs)
    num_non_empty = num_total - num_empty
    empty_sum = np.sum([d[0] for d in dice_coeffs if d[1] == 'empty'])
    non_empty_sum = np.sum([d[0] for d in dice_coeffs if d[1] == 'non-empty'])
    dice_coeff_empty = empty_sum / num_empty
    dice_coeff_non_empty = non_empty_sum / num_non_empty
    progress.write('[Empty: %d]: %.3f | %.3f, [Non-Empty: %d]: %.3f | %.3f' % (
        num_empty, dice_coeff_empty, empty_sum / num_total,
        num_non_empty, dice_coeff_non_empty, non_empty_sum / num_total))
    dice_coeff = float(dice_coeff)
    summary_str = 'f%02d-ep-%04d-val_dice-%.4f@%.2f' % (config.fold, epoch,
        dice_coeff, thresholds[best_threshold_ind])
    progress.write(summary_str)
    if dice_coeff > best_dice:
      weight_fn = os.path.join(config.logdir, summary_str + '.pth')
      th.save(model.state_dict(), weight_fn)
      best_dice = dice_coeff
      best_fn = weight_fn
      fns = sorted(glob(os.path.join(config.logdir, 'f%02d-*.pth' %
          config.fold)))
      for fn in fns[:-config.n_keep]:
        os.remove(fn)

  # create submission
  test_ds = DicomDataset(test_image_fns, height=config.height,
      width=config.height, write_cache=not config.is_kernel)
  test_loader = data.DataLoader(test_ds, batch_size=config.batch_size,
                               shuffle=False, num_workers=0,
                               pin_memory=False, drop_last=False)
  if best_fn is not None:
    model.load_state_dict(th.load(best_fn))
  model.eval()
  sub = create_submission(model, test_loader, test_image_fns, config,
      pred_zip=config.pred_zip)
  sub.to_csv(config.submission_fn, index=False)
  print("Wrote to: %s" % config.submission_fn)

  # create val submission
  val_fn = config.submission_fn.replace('.csv', '_VAL.csv')
  model.eval()
  sub = []
  sub = create_submission(model, val_loader, val_image_fns, config,
      pred_zip=config.pred_zip.replace('.zip', '_VAL.zip'))
  sub.to_csv(val_fn, index=False)
  print("Wrote to: %s" % val_fn)


class Config:
  def as_dict(self):
    return vars(self)
  def __str__(self):
    return str(self.as_dict())
  def __repr__(self):
    return str(self)


tic = time.time()
config = Config()
config.id = 37
config.train_dir = 'dicom-images-train'
config.test_dir = 'dicom-images-test'
config.sample_submission = 'sample_submission.csv'
config.train_rle = 'train-rle.csv'
config.epochs = 30
config.height = 512
config.batch_size = 32
config.lr = 1e-4
config.weight_decay = 0.0
config.weight = None  # 'PA_SEG_logdir_010_f00/ep-0012-val_dice-0.2812@0.30.pth'  # noqa
config.warmup = 0.05
config.accumulation_step = 1
config.num_folds = 5
config.num_workers = 0
config.p_clf = 0.6
config.p_seg = 0.3
config.pin = False
config.adjust_to_gt = False
config.slug = 'rx50'
config.device = 'cuda'
config.drop_empty = False
config.apex = True
config.view_position = 'Both'
config.n_keep = 1
config.is_kernel = True
for fold in [0, ]:  # range(5)
  config.fold = fold
  config.logdir = '%s_SEG_logdir_%03d_f%02d' % (config.view_position,
    config.id, config.fold)
  config.pred_zip = os.path.join(config.logdir, 'f%02d-PREDS.zip' % (
      config.fold))
  config.submission_fn = os.path.join(config.logdir,
      '%s_SEG_sub_%d_f%02d.csv' % (config.view_position, config.id,
        config.fold))
  print(config)
  main(config)
print("Duration: %.3f mins" % ((time.time() - tic) / 60))