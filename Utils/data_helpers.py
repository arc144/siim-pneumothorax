import pydicom
import numpy as np
import cv2
import torch
from Core.metrics import cmp_iou
from torch.utils.data._utils.collate import default_collate
import skimage.feature
import os

def read_image(path, color=True):
    assert os.path.exists(path), 'Path does not exist: {}'.format(path)
    if path.endswith('.dcm'):
        return read_dicom(path, color)
    else:
        im = cv2.imread(path)
        meta = dict(width=1024, height=1024)
        return im, meta

def read_dicom(path, color=True):
    dcm = pydicom.dcmread(path)
    # if dcm.file_meta.TransferSyntaxUID is None:
    #     dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    im = dcm.pixel_array
    if color:
        im = np.atleast_3d(im).repeat(3, 2)
    meta = dict(height=dcm.get('Rows'),
                width=dcm.get('Columns'),
                id=dcm.get('PatientID'),
                view=dcm.get('ViewPosition'),
                age=dcm.get("PatientAge"),
                sex=dcm.get("PatientSex"),
                )
    return im, meta

def input_lbp(im):
    im = im[:, :, 0]
    im = np.stack([im,
                   skimage.feature.local_binary_pattern(im, 8 * 9, 9, method='uniform'),
                   skimage.feature.local_binary_pattern(im, 8 * 11, 11, method='uniform')],
                  axis=2)
    return im

def mixup_data(x, y, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]
    # return mixed_x, y_a, y_b, lam
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


def resize_image(image, tgt_size):
    if tgt_size is None or image.shape == tgt_size:
        return image
    return cv2.resize(image, (tgt_size[0], tgt_size[1]))


def uint2float(im):
    im = np.asarray(im) / 255.
    return im


def toTensor(im):
    im = np.moveaxis(np.atleast_3d(im), -1, 0)
    im = torch.from_numpy(im).float()
    return im


def default_batch_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return [], [], [], []

    im_ids, meta, im, mask = [], [], [], []
    for b in batch:
        im_ids.append(b[0])
        meta.append(b[1])
        im.append(b[2])
        mask.append(b[3])

    im = torch.stack(im, dim=0)
    mask = torch.stack(mask, dim=0)
    return im_ids, meta, im, mask

def mask2rle(x):
    bs = np.where(x.T.flatten())[0]
    if len(bs) == 0:
        return '-1'

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1):
            rle.extend((int(b - np.sum(rle)), 0))

        rle[-1] += 1
        prev = b

    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    # if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
    #    rle[-2] = rle[-2] -1

    rle = ' '.join([str(r) for r in rle])
    return rle

# def mask2rle(img, width, height):
#     rle = []
#     lastColor = 0
#     currentPixel = 0
#     runStart = -1
#     runLength = 0
#
#     for x in range(width):
#         for y in range(height):
#             currentColor = img[x][y]
#             if currentColor != lastColor:
#                 if currentColor == 255:
#                     runStart = currentPixel
#                     runLength = 1
#                 else:
#                     rle.append(runStart)
#                     rle.append(runLength)
#                     runStart = -1
#                     runLength = 0
#                     currentPixel = 0
#             elif runStart > -1:
#                 runLength += 1
#             lastColor = currentColor
#             currentPixel += 1
#     if lastColor == 255:
#         rle.append(runStart)
#         rle.append(runLength)
#
#     return " ".join([str(r) for r in rle])


def rle2mask(rle, width, height, value=255):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = value
        current_position += lengths[index]

    return mask.reshape(width, height).T

