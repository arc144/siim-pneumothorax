import os
from glob import glob

exp_id = 3
prefix = 'Both_SEG_logdir_'
prefix = 'L2_logdir'

for fold in range(5):
  fn = '%s_%03d_f%02d' % (prefix, exp_id, fold)
  os.makedirs(fn, exist_ok=True)
  downloads = sorted(glob('/home/steffen/Downloads/*f%02d*' % fold))
  print("Fold %d: %d" % (fold, len(downloads)))
  for download in downloads:
    new_fn = os.path.join(fn, os.path.basename(download))
    os.rename(download, new_fn)
    os.system('sudo chmod -R +rwx %s' % new_fn)
