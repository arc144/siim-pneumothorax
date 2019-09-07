import torch.nn as nn
import torch.nn.functional as F
import torch
from .lovász_loss import lovasz_hinge, binary_xloss, lovasz_hinge_attention


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        loss = self.criterion(score.view(score.shape[0], 2, -1),
                              target.view(target.shape[0], -1).long())

        return loss


class OhemBCE(nn.Module):
    def __init__(self, thres=0.2,
                 min_kept=0.01, weight=None):
        super(OhemBCE, self).__init__()
        self.thresh = thres
        self.min_kept = min_kept
        self.criterion = nn.BCEWithLogitsLoss(weight=weight,
                                              reduction='none')

    def forward(self, score, target, **kwargs):
        score = score.view(-1, )
        target = target.view(-1, )

        pred = torch.abs(torch.sigmoid(score) - 0.5)
        mask = pred < self.thresh

        pixel_losses = self.criterion(score, target).contiguous().view(-1)

        pred, ind = pred[mask].sort()
        min_kept = int(self.min_kept * len(target))
        min_ix = min(min_kept, mask.sum())

        pixel_losses = pixel_losses[mask][ind[:min_ix]]
        return pixel_losses.mean()

class WeightedBCE(nn.Module):
    def __init__(self, pos_w=0.25, neg_w=0.75):
        super().__init__()
        self.neg_w = neg_w
        self.pos_w = pos_w

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

        return loss

################### DICE ########################
def dice_score2(logit, truth, smooth=2):
    prob = torch.sigmoid(logit)
    intersection = torch.sum(prob * truth)
    union = torch.sum(prob + truth)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def dice_score(logit, truth, smooth=2, mirror=False):
    logit = logit.view(logit.shape[0], -1)
    truth = truth.view(truth.shape[0], -1)

    if mirror:
        logit = 1 - logit
        truth = 1 - truth

    prob = torch.sigmoid(logit)
    intersection = torch.sum(prob * truth, dim=1)
    union = torch.sum(prob + truth, dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, mirror=False):
        super(DiceLoss, self).__init__()
        self.mirror = mirror
        self.smooth = smooth

    def forward(self, logit, truth):
        dice = dice_score(logit, truth, self.smooth, self.mirror)
        loss = 1 - dice
        return loss


################ FOCAL LOSS ####################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=500):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return self.alpha * loss.mean()


class Focal_LogDice(nn.Module):
    def __init__(self, alpha=10, gamma=2):
        super().__init__()
        self.focal = FocalLoss(gamma, alpha)

    def forward(self, input, target):
        fl = self.focal(input, target)
        logdice = - torch.log(dice_score(input, target))
        loss = fl + logdice
        return loss.mean()


################# BCE + DICE ########################
class BCE_Dice(nn.Module):
    def __init__(self, smooth=1, alpha=0.3):
        super(BCE_Dice, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        #self.dice = DiceLoss(smooth=smooth)
        #self.bce = nn.BCEWithLogitsLoss()

#    def forward(self, logit, truth):
#        dice = self.dice(logit, truth)
#        bce = self.bce(logit, truth)
#        return self.alpha * dice + bce


    def forward(self, y_pred, y_true, weights=None, dice_weight=0.04):
        assert y_true.shape == y_pred.shape
        y_pred = torch.sigmoid(y_pred)
        y_true_sum = y_true.sum((1, 2, 3))
        non_empty = torch.gt(y_true_sum, 0)
        total_loss = 0.0
        # dice loss
        if non_empty.sum() > 0:
            yt_non_empty, yp_non_empty = y_true[non_empty], y_pred[non_empty]
            intersection = (yt_non_empty * yp_non_empty).sum((1, 2, 3))
            dice = (2. * intersection) / (yt_non_empty.sum((1, 2, 3)) +
                yp_non_empty.sum((1, 2, 3)))
            dl = torch.mean(1. - dice)
            total_loss += dice_weight * dl

        # bce loss
        y_pred = torch.clamp(y_pred, 1e-6, 1. - 1e-6)
        bce = -y_true * torch.log(y_pred) - (1. - y_true) * torch.log(1. - y_pred)
        bce = torch.mean(bce)
        total_loss += bce

        return total_loss


############### LOVÁSZ-HINGE ########################
class Lovasz_Hinge(nn.Module):
    def __init__(self, per_image=True):
        super(Lovasz_Hinge, self).__init__()
        self.per_image = per_image

    def forward(self, logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = lovasz_hinge(logit, truth,
                            per_image=self.per_image)
        return loss


############## BCE + LOVÁSZ #########################
class BCE_Lovasz(nn.Module):
    def __init__(self, per_image=True, alpha=1):
        super(BCE_Lovasz, self).__init__()
        self.alpha = alpha
        self.per_image = per_image

    def forward(self, logit, truth):
        bce = binary_xloss(logit, truth)

        pos_ix = (truth.view(truth.shape[0], -1).sum(1) > 0)
        lovasz = lovasz_hinge(logit[pos_ix], truth[pos_ix], per_image=self.per_image)
        return bce + self.alpha * lovasz


############## ANGULAR MSE #########################
class AngularMSE(nn.Module):
    def __init__(self):
        super(AngularMSE, self).__init__()

    def forward(self, gt_flat_mask, pred_vec, gt_vec):
        batch = gt_flat_mask.shape[0]
        ix = gt_flat_mask.view(batch, -1).byte()
        if ix.sum() == 0:
            return 0

        pred_vec_x = pred_vec[:, 0, ...].view(batch, -1)[ix]
        gt_vec_x = gt_vec[:, 0, ...].view(batch, -1)[ix]

        pred_vec_y = pred_vec[:, 1, ...].view(batch, -1)[ix]
        gt_vec_y = gt_vec[:, 1, ...].view(batch, -1)[ix]

        return torch.mean((gt_vec_x - pred_vec_x) ** 2 + (gt_vec_y - pred_vec_y) ** 2)
