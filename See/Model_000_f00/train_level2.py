import torch as th
import torch.utils.data as data
from tqdm import tqdm
import os
import time
import apex
from matplotlib import pyplot as plt  # noqa
import numpy as np
from glob import glob

from segmentation_model import FPNSegmentation
from data import L2DicomDataset, load_gt
from schedules import WarmupLinearSchedule
from server import score
from submit_segmentation import create_submission


SEED = 32
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def siim_loss(y_true, y_pred, weights=None, dice_weight=0.3):
  assert y_true.shape == y_pred.shape
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
  os.makedirs('cache', exist_ok=True)
  os.makedirs(config.logdir, exist_ok=True)
  print("Logging to: %s" % config.logdir)
  if not os.path.exists(config.train_dir):
    print("KERNEL ENV")
    config.train_dicom_dir = '../input/siim-train-test/siim/dicom-images-train'
    config.test_dicom_dir = '../input/siim-train-test/siim/dicom-images-test'

    config.train_dir = '../input/l2-images/l2-images/l2-images-train'
    config.test_dir = '../input/l2-images/l2-images/l2-images-test'

    config.sample_submission = '../input/siim-acr-pneumothorax-segmentation/' \
        'sample_submission.csv'
    config.train_rle = '../input/siim-train-test/siim/train-rle.csv'

  train_image_fns = sorted(glob(os.path.join(config.train_dir, '*.png')))
  test_image_fns = sorted(glob(os.path.join(config.test_dir, '*.png')))

  # assert len(train_image_fns) == 10675, len(train_image_fns)
  # assert len(test_image_fns) in (1372, 1377), len(test_image_fns)

  gt = load_gt(config.train_rle)
  # create folds
  if not config.stratify:
    # random folds
    np.random.shuffle(train_image_fns)
  else:
    # folds stratified by mask size
    train_mask_sizes = [L2DicomDataset.rles_to_mask(gt[
        L2DicomDataset.fn_to_id(fn)]).sum() for fn in tqdm(train_image_fns)]
    sorted_inds = [k for k in sorted(range(len(train_image_fns)),
        key=lambda k: train_mask_sizes[k])]
    train_image_fns = [train_image_fns[k] for k in sorted_inds]

  folds = np.arange(len(train_image_fns)) % config.num_folds
  val_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] == config.fold]
  train_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] != config.fold]
  # remove not-used files:
  # https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98478#latest-572385  # noqa
  train_image_fns = [fn for fn in train_image_fns
      if L2DicomDataset.fn_to_id(fn) in gt]
  val_image_fns = [fn for fn in val_image_fns
      if L2DicomDataset.fn_to_id(fn) in gt]

  if config.drop_empty:
    # remove empty masks from training data
    non_empty_gt = {k: v for k, v in gt.items() if v[0] != ' -1'}
    train_image_fns = [fn for fn in train_image_fns if
        L2DicomDataset.fn_to_id(fn) in non_empty_gt]
    print("[Non-EMPTY] TRAIN: ", len(train_image_fns), os.path.basename(
        train_image_fns[0]))

  print("VAL: ", len(val_image_fns), os.path.basename(val_image_fns[0]))
  print("TRAIN: ", len(train_image_fns), os.path.basename(train_image_fns[0]))

  train_ds = L2DicomDataset(train_image_fns, gt_rles=gt, height=config.height,
      width=config.height, to_ram=True, augment=True,
      write_cache=not config.is_kernel,
      train_dicom_dir=config.train_dicom_dir,
      test_dicom_dir=config.test_dicom_dir)
  val_ds = L2DicomDataset(val_image_fns, gt_rles=gt, height=config.height,
      width=config.height, to_ram=True,
      write_cache=not config.is_kernel,
      train_dicom_dir=config.train_dicom_dir,
      test_dicom_dir=config.test_dicom_dir)

  val_loader = data.DataLoader(val_ds, batch_size=config.batch_size,
                               shuffle=False, num_workers=config.num_workers,
                               pin_memory=config.pin, drop_last=False)

  model = FPNSegmentation(config.slug, num_input_channels=2)
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
      accuracy = th.mean(((y_pred >= 0.5) == (y_true >= 0.5)).to(
          th.float)).item()
      smooth_accuracy = accuracy if smooth_accuracy is None else \
          smooth * accuracy + (1. - smooth) * smooth_accuracy
      progress.set_postfix(loss='%.4f' % smooth_loss, accuracy='%.4f' %
            (smooth_accuracy), lr='%.6f' % (config.lr if lr_this_step is None
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
          yp = yp.squeeze().cpu().numpy()
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
  test_ds = L2DicomDataset(test_image_fns, height=config.height,
      width=config.height, write_cache=not config.is_kernel,
      train_dicom_dir=config.train_dicom_dir,
      test_dicom_dir=config.test_dicom_dir)
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


if __name__ == '__main__':
  tic = time.time()
  config = Config()
  config.id = 2
  config.train_dir = 'l2-images-train'
  config.test_dir = 'l2-images-test'

  config.train_dicom_dir = 'dicom-images-train'
  config.test_dicom_dir = 'dicom-images-test'

  config.sample_submission = 'sample_submission.csv'
  config.train_rle = 'train-rle.csv'
  config.epochs = 20
  config.height = 512  # TODO
  config.batch_size = 16
  config.lr = 1e-4
  config.weight_decay = 0.0
  config.weight = None  # 'PA_SEG_logdir_010_f00/ep-0012-val_dice-0.2812@0.30.pth'  # noqa
  config.warmup = 0.05
  config.accumulation_step = 1
  config.num_folds = 5
  config.num_workers = 4
  config.p_clf = 0.6
  config.p_seg = 0.2
  config.pin = False
  config.adjust_to_gt = False
  config.slug = 'r34'
  config.device = 'cuda'
  config.drop_empty = False
  config.apex = True
  config.n_keep = 1
  config.stratify = False
  config.is_kernel = False
  for fold in [0, ]:  # range(5)
    config.fold = fold
    config.logdir = 'L2_logdir_%03d_f%02d' % (config.id, config.fold)
    config.pred_zip = os.path.join(config.logdir, 'f%02d-PREDS.zip' % (
        config.fold))
    config.submission_fn = os.path.join(config.logdir,
        'L2_sub_%d_f%02d.csv' % (config.id,
          config.fold))
    print(config)
    main(config)
  print("Duration: %.3f mins" % ((time.time() - tic) / 60))
