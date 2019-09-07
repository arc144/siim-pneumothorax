import pandas as pd
import os
import numpy as np


def invert_pred(df, new_true='1 2'):
    ix_no_ptx = df[df['EncodedPixels'] == '-1'].index
    ix_ptx = df[df['EncodedPixels'] != '-1'].index

    df.loc[ix_no_ptx, 'EncodedPixels'] = new_true
    df.loc[ix_ptx, 'EncodedPixels'] = '-1'
    return df


if __name__ == '__main__':
    CSV_PATH = '/media/hdd/Kaggle/Pneumothorax/Output/best_sub.csv'
    df = pd.read_csv(CSV_PATH)
    df = invert_pred(df)
    df.to_csv(CSV_PATH.replace('.csv', '_inverted.csv'), index=False)
