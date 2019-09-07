import numpy as np
from skimage.measure import label
import pandas as pd
from Utils.data_helpers import mask2rle


def split_instances(mask, bin_ths=0.5, connectivity=2, min_size=0, out_value=255):
    B, H, W = mask.shape
    bin_mask = (mask > bin_ths).astype(np.int)
    out = []
    for i in range(B):
        pred_i, num = label(bin_mask[i], connectivity=connectivity, return_num=True)
        pred_i_split = []
        # Remove small labels
        for j in range(1, num + 1):
            color_j = (pred_i == j).astype(np.int) * out_value
            area_j = color_j.sum()
            if area_j > min_size:
                pred_i_split.append(color_j)
        out.append(np.array(pred_i_split) if len(pred_i_split) > 0 else np.zeros((1, H, W)))
    return out

def make_submission(index_vec, pred_vec, output, width=1024, height=1024):
    ImageIds, rles = [], []
    # Multiple ids in pred_vec
    for i, (id, pred) in enumerate(zip(index_vec, pred_vec)):
        # Multiple preds per id
        # for p in pred:
        rle = mask2rle(pred)
        ImageIds.append(id)
        rles.append(rle)
    df = pd.DataFrame(dict(ImageId=ImageIds, EncodedPixels=rles))
    df.to_csv(output, index=False)

def count_connected_instances(mask, ths=0.2, connectivity=2):
    B, H, W = mask.shape
    out = []
    bin_mask = (mask > ths).astype(np.uint8)
    for i in range(B):
        _, num = label(bin_mask[i], connectivity=connectivity, return_num=True)
        out.append(num)
    return out