import pandas as pd
import pickle
from glob import glob
import os
from datetime import datetime


from data import split_by_view, DicomDataset  # noqa


os.makedirs('cache', exist_ok=True)
train_image_fns = sorted(glob(os.path.join('dicom-images-train', '*/*/*.dcm')))
test_image_fns = sorted(glob(os.path.join('dicom-images-test', '*/*/*.dcm')))

cache_fn = 'cache/view.p'
if not os.path.exists(cache_fn):
  train_image_fns_PA, train_image_fns_AP = split_by_view(train_image_fns)
  test_image_fns_PA, test_image_fns_AP = split_by_view(test_image_fns)
  with open(cache_fn, 'wb') as f:
    pickle.dump([train_image_fns_PA, train_image_fns_AP,
        test_image_fns_PA, test_image_fns_AP], f)
else:
  with open(cache_fn, 'rb') as f:
    train_image_fns_PA, train_image_fns_AP, test_image_fns_PA, \
        test_image_fns_AP = pickle.load(f)


gt = pd.read_csv('train-rle.csv')
# merge multi-rle
gt = gt.groupby('ImageId', as_index=False).agg(lambda x: x.iloc[0])
train_image_fns_PA, train_image_fns_AP, test_image_fns_PA, \
    test_image_fns_AP = map(set, (train_image_fns_PA, train_image_fns_AP,
          test_image_fns_PA, test_image_fns_AP))
train_image_ids_PA, train_image_ids_AP, test_image_ids_PA, \
    test_image_ids_AP = map(lambda x: [DicomDataset.fn_to_id(ele) for ele in x],
        (train_image_fns_PA, train_image_fns_AP,
            test_image_fns_PA, test_image_fns_AP))
gt['View'] = gt['ImageId'].apply(lambda x: 'PA' if x in train_image_ids_PA
    else 'AP')

print(gt.groupby('View')[' EncodedPixels'].agg(lambda x: (x == ' -1').mean()))
print("MASK SAMPLES: ", gt.groupby('View')[' EncodedPixels'].agg(
    lambda x: (x != ' -1').sum()))

print("TRAIN: ", gt.groupby('View')[' EncodedPixels'].agg(lambda x: len(x)))
print("TEST PA: %d, AP: %d" % (len(test_image_fns_PA), len(test_image_fns_AP)))


print("Train Fraction: %.3f" % (len(train_image_ids_AP) / len(
    train_image_ids_PA)))
print("Test Fraction: %.3f" % (len(test_image_ids_AP) / len(
    test_image_ids_PA)))


gt = pd.read_csv('train-rle.csv')
gt['time'] = gt['ImageId'].apply(lambda x: str(datetime.utcfromtimestamp(
    float('.'.join(x.split('.')[-2:])))))
gt['has_mask'] = gt[' EncodedPixels'].apply(lambda x: '-1' not in x)
gt = gt.sort_values('time')
gt['magic'] = gt['ImageId'].apply(lambda x: str(x.split('.')[-3]))
gt = gt.sort_values('magic')
gt_times = set(gt['time'].unique())
gt_ids = set(gt['ImageId'].unique())

mask_percentage = gt.groupby('time')['has_mask'].agg(
    lambda x: (x.mean(), len(x))).sort_index()
test = pd.DataFrame({'ImageId': [DicomDataset.fn_to_id(fn)
    for fn in test_image_fns]})


test = pd.read_csv('sample_submission_leak.csv')
test['time'] = test['ImageId'].apply(lambda x: str(datetime.utcfromtimestamp(
    float('.'.join(x.split('.')[-2:])))))
test['magic'] = test['ImageId'].apply(lambda x: str(x.split('.')[-3]))
test = test.sort_values('magic')
test_p = test.groupby('time').agg(lambda x: len(x)).sort_index()


ll = pd.read_csv('sample_submission_leak.csv')
ll['time'] = ll['ImageId'].apply(lambda x: str(datetime.utcfromtimestamp(
    float('.'.join(x.split('.')[-2:])))))
ll['magic'] = ll['ImageId'].apply(lambda x: str(x.split('.')[-3]))
ll = ll.sort_values('magic')
