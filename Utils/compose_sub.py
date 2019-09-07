import pandas as pd
import os
import numpy as np


def compose_sub(cls_path, seg_path):
    df_cls = pd.read_csv(CLS_PATH)
    df = pd.read_csv(SEG_PATH)
    df.loc[df_cls['EncodedPixels'] == '-1', 'EncodedPixels'] = '-1'
    return df


if __name__ == '__main__':
    SEG_PATH = '/media/hdd/Kaggle/Pneumothorax/Output/ens_seg_with_see.csv'
    CLS_PATH = '/media/hdd/Kaggle/Pneumothorax/Output/ens_cls_with_see2.csv'

    df = compose_sub(CLS_PATH, SEG_PATH)
    print('{} non-empty masks'.format(len(df[df['EncodedPixels'] != '-1'])))
    df.to_csv(SEG_PATH.replace('.csv', '_composed.csv'), index=False)
