import numpy as np
from scipy.optimize import linear_sum_assignment


def cmp_iou(pred, gt, eps=1e-12):
    bin_pred = (pred > 0.5).reshape(pred.shape[0], -1)
    bin_gt = (gt > 0.5).reshape(gt.shape[0], -1)

    overlap = np.logical_and(bin_pred, bin_gt).sum(1)
    union = np.logical_or(bin_pred, bin_gt).sum(1) + eps

    iou = overlap / union

    ix = np.where(bin_gt.sum(1) == 0)[0]
    iou[ix] = (bin_pred.sum(1)[ix] == 0).astype(np.float)
    return iou


def cmp_dice(pred, gt, eps=1e-12):
    bin_pred = pred.reshape(pred.shape[0], -1)
    bin_gt = (gt > 0.5).reshape(gt.shape[0], -1)

    overlap = (bin_pred * bin_gt).sum(1)
    union = bin_pred.sum(1) + bin_gt.sum(1) + eps

    dice = 2 * overlap / union
    ptx_dice = dice[bin_gt.sum(1) > 0]
    ix = np.where(bin_gt.sum(1) == 0)[0]
    dice[ix] = (bin_pred.sum(1)[ix] == 0).astype(np.float)
    return dice, ptx_dice


def cmp_cls_acc(pred, gt):
    pred_cls = (pred.reshape(pred.shape[0], -1).sum(1) > 0)
    gt_cls = (gt.reshape(gt.shape[0], -1).sum(1) > 0)
    acc = (pred_cls == gt_cls).astype(np.uint8)
    return acc


def cmp_instance_dice(instance_preds, instance_targs):
    '''
    instance dice score
    instance_preds: list of N_i mask (0,1) per image - variable preds per image
    instance_targs: list of M_i mask (0,1) target per image - variable targs per image
    '''

    #     # for plotting
    #     fig,axes=plt.subplots(1,2)
    #     axes[0].imshow(np.concatenate(valid_instance_targs[i]))
    #     axes[1].imshow(np.concatenate(valid_instance_preds[i]))

    scores = []
    for i in range(len(instance_preds)):
        # Case when there is no GT mask
        if np.sum(instance_targs[i]) == 0:
            scores.append(int(np.sum(instance_preds[i]) == 0))
        # Case when there is no pred mask but there is GT mask
        elif np.sum(instance_preds[i]) == 0:
            scores.append(0)
        # Case when there is both pred and gt masks
        else:
            m, _, _ = instance_targs[i].shape
            n, _, _ = instance_preds[i].shape

            targs = instance_targs[i].reshape(m, -1)
            preds = instance_preds[i].reshape(n, -1)

            # intersect: matrix of targ x preds (M, N)
            intersect = ((targs[:, None, :] * preds[None, :, :]) > 0).sum(2)
            targs_area, preds_area = targs.sum(1), preds.sum(1)
            union = targs_area[:, None] + preds_area[None, :]

            dice = (2 * intersect / union)

            dice_scores = dice[linear_sum_assignment(1 - dice)]
            mean_dice_score = sum(dice_scores) / max(n, m)  # unmatched gt or preds are counted as 0
            scores.append(mean_dice_score)
    return scores
