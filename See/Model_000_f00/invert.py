import pandas as pd
import os

sub_fn = 'x_ensemble/CLF_ADJUST_Both_ENS_V3_0054.csv'
out_fn = 'INVERT_' + os.path.basename(sub_fn)

sub = pd.read_csv(sub_fn).groupby('ImageId', as_index=False).agg(
    lambda x: x.iloc[0])
sub['EncodedPixels'] = sub['EncodedPixels'].apply(
    lambda x: '1 1' if '-1' in x else '-1')
sub.to_csv(out_fn, index=False)
print("Wrote to: %s" % out_fn)
