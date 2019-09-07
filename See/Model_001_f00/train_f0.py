import torch as th
import torch.utils.data as data
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
import os
import time
import apex
import pandas as pd
from shutil import copyfile
from matplotlib import pyplot as plt  # noqa
import numpy as np
from glob import glob

from segmentation_model import FPNSegmentation
from data import DicomDataset, load_gt
from schedules import WarmupLinearSchedule
from server import score
from submit_segmentation import create_submission
from adamw import AdamW

th.backends.cudnn.benchmark = True


def seed_all():
  SEED = 123
  np.random.seed(SEED)
  th.manual_seed(SEED)
  th.cuda.manual_seed(SEED)


def siim_loss(y_true, y_pred, weights=None, dice_weight=0.3):
  assert y_true.shape == y_pred.shape
  y_true = y_true.to(y_pred.dtype)
  y_true_sum = y_true.sum((1, 2, 3))
  non_empty = th.gt(y_true_sum, 0)
  total_loss = 0.0
  # dice loss
  if non_empty.sum() > 0:
    yt_non_empty, yp_non_empty = y_true[non_empty], y_pred[non_empty]
    intersection = (yt_non_empty * yp_non_empty).sum((1, 2, 3))
    dice = (2. * intersection) / (yt_non_empty.sum((1, 2, 3)) +
        yp_non_empty.sum((1, 2, 3)))
    dl = th.mean(1. - dice)
    total_loss += dice_weight * dl

  # bce loss
  y_pred = th.clamp(y_pred, 1e-6, 1. - 1e-6)
  bce = -y_true * th.log(y_pred) - (1. - y_true) * th.log(
      1. - y_pred)
  bce = th.mean(bce)
  total_loss += bce

  return total_loss


def main(config):
  seed_all()
  os.makedirs('cache', exist_ok=True)
  os.makedirs(config.logdir, exist_ok=True)
  print("Logging to: %s" % config.logdir)
  src_files = sorted(glob('*.py'))
  for src_fn in src_files:
    dst_fn = os.path.join(config.logdir, src_fn)
    copyfile(src_fn, dst_fn)

  train_image_fns = sorted(glob(os.path.join(config.train_dir, '*/*/*.dcm')))
  test_image_fns = sorted(glob(os.path.join(config.test_dir, '*/*/*.dcm')))

  assert len(train_image_fns) == 10712
  assert len(test_image_fns) == 1377

  gt = load_gt(config.train_rle)
  # create folds
  np.random.shuffle(train_image_fns)

  if config.subset > 0:
    train_image_fns = train_image_fns[:config.subset]

  folds = np.arange(len(train_image_fns)) % config.num_folds
  val_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] == config.fold]
  train_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] != config.fold]
  # remove not-used files:
  # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98478#latest-572385  # noqa
  train_image_fns = [fn for fn in train_image_fns
      if DicomDataset.fn_to_id(fn) in gt]
  val_image_fns = [fn for fn in val_image_fns
      if DicomDataset.fn_to_id(fn) in gt]

  print("VAL: ", len(val_image_fns), os.path.basename(val_image_fns[0]))
  print("TRAIN: ", len(train_image_fns), os.path.basename(train_image_fns[0]))

  train_ds = DicomDataset(train_image_fns, gt_rles=gt, height=config.height,
      width=config.height, augment=True)
  val_ds = DicomDataset(val_image_fns, gt_rles=gt, height=config.height,
      width=config.height)

  if config.cache:
    train_ds.cache()
    val_ds.cache()

  val_loader = data.DataLoader(val_ds, batch_size=config.batch_size,
                               shuffle=False, num_workers=config.num_workers,
                               pin_memory=config.pin, drop_last=False)

  model = FPNSegmentation(config.slug, ema=config.ema)
  if config.weight is not None:
    print("Loading: %s" % config.weight)
    model.load_state_dict(th.load(config.weight))
  model = model.to(config.device)

  no_decay = ['mean', 'std', 'bias'] + ['.bn%d.' % i for i in range(100)]
  grouped_parameters = [{'params': [], 'weight_decay': config.weight_decay},
      {'params': [], 'weight_decay': 0.0}]
  for n, p in model.named_parameters():
    if not any(nd in n for nd in no_decay):
      print("Decay: %s" % n)
      grouped_parameters[0]['params'].append(p)
    else:
      print("No Decay: %s" % n)
      grouped_parameters[1]['params'].append(p)
  optimizer = AdamW(grouped_parameters, lr=config.lr)

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
  for epoch in range(1, config.epochs + 1):
    smooth_loss = None
    smooth_accuracy = None
    model.train()
    train_loader = data.DataLoader(train_ds, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.num_workers,
                                   pin_memory=config.pin, drop_last=True)
    progress = tqdm(total=len(train_ds), smoothing=0.01)
    for i, (X, _, y_true) in enumerate(train_loader):
      X = X.to(config.device).float()
      y_true = y_true.to(config.device)
      y_pred = model(X)
      loss = siim_loss(y_true, y_pred, weights=None)
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
      accuracy = th.mean(((y_pred >= 0.5) == (y_true == 1)).to(
          th.float)).item()
      smooth_accuracy = accuracy if smooth_accuracy is None else \
          smooth * accuracy + (1. - smooth) * smooth_accuracy
      progress.set_postfix(ep='%d/%d' % (epoch, config.epochs),
            loss='%.4f' % smooth_loss, accuracy='%.4f' %
            (smooth_accuracy), lr='%.6f' % (config.lr if lr_this_step is None
              else lr_this_step))
      progress.update(len(X))

    if epoch <= 10:
      continue
    # validation loop
    model.eval()
    thresholds = [0.1, 0.2]
    dice_coeffs = [[] for _ in range(len(thresholds))]
    progress = tqdm(enumerate(val_loader), total=len(val_loader))
    with th.no_grad():
      for i, (X, _, y_trues) in progress:
        X = X.to(config.device).float()
        y_trues = y_trues.to(config.device)
        y_preds = model(X)
        if epoch >= 12:
          y_preds_flip = th.flip(model(th.flip(X, (-1, ))), (-1, ))
          y_preds = 0.5 * (y_preds + y_preds_flip)

        y_trues = y_trues.cpu().numpy()
        y_preds = y_preds.cpu().numpy()
        for yt, yp in zip(y_trues, y_preds):
          yt = (yt.squeeze() >= 0.5).astype('uint8')
          yp = yp.squeeze()
          for dind, threshold in enumerate(thresholds):
            yp_ = (yp >= threshold).astype(np.uint8)
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
      width=config.height)
  test_loader = data.DataLoader(test_ds, batch_size=config.batch_size,
                               shuffle=False, num_workers=0,
                               pin_memory=False, drop_last=False)
  if best_fn is not None:
    model.load_state_dict(th.load(best_fn))
  model.eval()
  sub = create_submission(model, test_loader, config, pred_zip=config.pred_zip)
  sub.to_csv(config.submission_fn, index=False)
  print("Wrote to: %s" % config.submission_fn)

  # create val submission
  val_fn = config.submission_fn.replace('.csv', '_VAL.csv')
  model.eval()
  sub = []
  sub = create_submission(model, val_loader, config,
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


if __name__ == '__main__':
  tic = time.time()
  config = Config()
  config.id = 3
  config.train_dir = 'dicom-images-train'
  config.test_dir = 'dicom-images-test'
  config.sample_submission = 'sample_submission.csv'
  config.train_rle = 'train-rle.csv'
  config.epochs = 20
  config.height = 512
  config.batch_size = 16
  config.lr = 1e-4
  config.weight_decay = 1e-4
  config.weight = None
  config.warmup = 0.05
  config.accumulation_step = 1
  config.num_folds = 5
  config.num_workers = 4
  config.p_clf = 0.6
  config.p_seg = 0.2
  config.pin = False
  config.slug = 'r50d'  # TODO
  config.device = 'cuda'
  config.apex = False
  config.subset = -1  # TODO
  config.view_position = 'Both'
  config.n_keep = 1
  config.is_kernel = False
  config.ema = False
  config.cache = True  # TODO

  config.fold = 0  # 0
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
