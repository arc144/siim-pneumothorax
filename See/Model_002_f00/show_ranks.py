import pandas as pd
from matplotlib import pyplot as plt


train = pd.read_csv('train-rle.csv')
train['EncodedPixels'] = train[' EncodedPixels']
sub = pd.read_csv('x_ensemble/CLF_ADJUST_Both_ENS_V3_0039.csv')
sub = {k: v for k, v in zip(sub['ImageId'], sub['EncodedPixels'])}
sample_sub = pd.read_csv('sample_submission.csv')
sample_sub['EncodedPixels'] = sample_sub['ImageId'].apply(
    lambda k: sub.get(k, '-1'))

print(sample_sub.shape)
ordered_counts = []
current_count = 0
df = train
processed_ids = set()
for k, (img_id, rle) in enumerate(zip(df['ImageId'],
        df['EncodedPixels'])):
  if img_id in processed_ids:
    continue
  else:
    processed_ids.add(img_id)
  if '-1' not in rle:
    current_count += 1
  if k % 50 == 0:
    ordered_counts.append(current_count)
    current_count = 0

ordered_counts.append(current_count)

plt.bar(range(len(ordered_counts)), ordered_counts)
plt.show()
