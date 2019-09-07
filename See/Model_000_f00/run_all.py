from glob import glob
import os


# rsync -r --progress --exclude "*.zip" --exclude "*.pth" --exclude "*.pth" ../SIIM/Both_SEG_logdir_{088,089,095,104,105}_f00 .  # noqa
def run_cmd(cmd):
  print("Running: `%s`" % cmd)
  os.system(cmd)


def copy_file(src, dst, fold):
  with open(src, 'r') as f:
    lines = f.readlines()

  for k in range(len(lines)):
    if 'config.fold = ' in lines[k]:
      lines[k] = lines[k].replace('config.fold = ',
          'config.fold = %d  # ' % fold)
  with open(dst, 'w') as f:
    f.writelines(lines)


def main():
  models = [
    # 'model_03',
    'model_04',
  ]

  for model in models:
    for fold in range(5):
      all_py_fns = sorted(glob(os.path.join(model, '*.py')))
      dst_fns = []
      for fn in all_py_fns:
        dst_fn = os.path.basename(fn)
        if dst_fn == 'train.py':
          dst_fn = dst_fn.replace('.py', '_f%d.py' % fold)
        dst_fns.append(dst_fn)
        copy_file(fn, dst_fn, fold)

      cmd = "python3 train_f%d.py" % fold
      run_cmd(cmd)
      for fn in dst_fns:
        os.remove(os.path.basename(fn))

  cmd = "python3 ensemble_all_095.py"
  run_cmd(cmd)


if __name__ == '__main__':
  main()
