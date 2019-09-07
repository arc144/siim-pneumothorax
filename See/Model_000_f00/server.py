import pandas as pd  # noqa
import numpy as np
import argparse
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
from concurrent.futures import ThreadPoolExecutor
from data import DicomDataset, load_gt


def score(yt, yp):
  assert yt.dtype in ('uint8', 'int32'), yt.dtype
  assert yp.dtype in ('uint8', 'int32'), yp.dtype
  assert yt.shape == (1024, 1024), yt.shape
  assert yt.shape == yp.shape, yp.shape
  assert yt.max() <= 1, yt.max()
  assert yp.max() <= 1, yp.max()
  yt = (yt == 1)
  yp = (yp == 1)
  yt_sum = yt.sum()
  yp_sum = yp.sum()
  if yt_sum == 0:
    if yp_sum != 0:
      score = (0, 'empty', 'non-empty')
    else:
      score = (1, 'empty', 'empty')
    return score

  intersection = np.logical_and(yt, yp).sum()
  dice_coeff = (2 * intersection) / (yt_sum + yp_sum)
  score = (dice_coeff, 'non-empty',
      'empty' if yp_sum == 0 else 'non-empty')
  return score


def run_server(prediction_fn, gt_fn):
  submission = load_gt(prediction_fn, rle_key='EncodedPixels')
  gt = load_gt(gt_fn)

  def compute_score(key):
    yt = DicomDataset.rles_to_mask(gt[key], merge_masks=True)
    yp = DicomDataset.rles_to_mask(submission[key], merge_masks=True)
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


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', type=str)
  args = parser.parse_args()
  final_score = run_server(args.fn, 'train-rle.csv')
  print(round(final_score, 4))


if __name__ == '__main__':
  main()


# def score_v2(yt, yp):
#   assert yt.dtype == 'int32', yt.dtype
#   assert yp.dtype == 'int32', yp.dtype
#   assert yt.shape == (1024, 1024), yt.shape
#   assert yt.shape == yp.shape, yp.shape
#   num_gt_masks = yt.max()
#   num_pred_masks = yp.max()
#   if num_gt_masks == 0:
#     if num_pred_masks != 0:
#       score = (0, 'empty', 'non-empty')
#     else:
#       score = (1, 'empty', 'empty')
#     return score
#   per_image_scores = []
#   matched_pred_indices = []
#   for gt_index in range(1, num_gt_masks + 1):
#     gt_mask = yt == gt_index
#     best_dice_coeff = 0.
#     best_pred_index = None
#     for pred_index in range(1, num_pred_masks + 1):
#       if pred_index in matched_pred_indices:
#         continue
#       pred_mask = yp == pred_index
#       intersection = np.logical_and(gt_mask, pred_mask).sum()
#       dice_coeff = (2 * intersection) / (gt_mask.sum() + pred_mask.sum())
#       if dice_coeff > best_dice_coeff:
#         best_dice_coeff = dice_coeff
#         best_pred_index = pred_index

#     matched_pred_indices.append(best_pred_index)
#     per_image_scores.append(best_dice_coeff)

#   # too many predictions
#   per_image_scores.extend([0] * (num_pred_masks - len(matched_pred_indices)))
#   score = (np.mean(per_image_scores), 'non-empty',
#       'empty' if num_gt_masks == 0 else 'non-empty')
#   return score
