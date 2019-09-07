import os

fns = [
  'Both_SEG_logdir_088_f00',
  'Both_SEG_logdir_088_f01',
  'Both_SEG_logdir_088_f02',
  'Both_SEG_logdir_088_f03',
  'Both_SEG_logdir_088_f04',

  'Both_SEG_logdir_089_f00',
  'Both_SEG_logdir_089_f01',
  'Both_SEG_logdir_089_f02',
  'Both_SEG_logdir_089_f03',
  'Both_SEG_logdir_089_f04',
]

for fn in fns:
  fold = fn.split('_')[-1]
  os.system('rsync -r --progress see@192.168.0.234:/home/see/PyTorch/SIIM'
      '/%s/%s-TTA-PREDS*.zip %s/' % (fn, fold, fn))
