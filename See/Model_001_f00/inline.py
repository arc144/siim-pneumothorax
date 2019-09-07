import os

custom_modules = ['data', 'segmentation_model', 'schedules', 'server',
    'submit_segmentation']
script = 'train_level2.py'
os.makedirs('build', exist_ok=True)

with open(script, 'r') as f:
  content = f.readlines()


def is_import(l, m):
  return l.startswith('from %s' % m) or l.startswith('import %s' % m)


inlined_content = []
for line in content:
  inlined_line = False
  if any(is_import(line, m) for m in custom_modules):
    module = [m for m in custom_modules if is_import(line, m)][0]
    with open('%s.py' % module) as f:
      for inline_line in f:
        if '__main__' in inline_line or inline_line.startswith('def main('):
          break
        if any(inline_line.startswith('from %s' % m) for m in custom_modules):
          continue
        inlined_content.append(inline_line)
    inlined_line = True
  if not inlined_line:
    inlined_content.append(line)

out_fn = os.path.join('build', script)
indent = 0
with open(out_fn, 'w') as f:
  for line in inlined_content:
    if line == 'from tqdm import tqdm\n':
      line = 'from tqdm import tqdm_notebook as tqdm\n'
    if line == "if __name__ == '__main__':\n":
      indent = 2
      continue
    if 'is_kernel = False' in line:
      line = line.replace('is_kernel = False', 'is_kernel = True')
    if 'IPython' in line:
      print("Possible bug: ", line[:-1])
      raise RuntimeError()
    f.write(line[indent:])

print("Wrote script to: %s" % out_fn)
os.system('subl %s' % out_fn)
