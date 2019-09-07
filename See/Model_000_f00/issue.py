import torch as th
from segmentation_model import FPNSegmentation
import cv2
import pydicom
from matplotlib import pyplot as plt

gold = cv2.imread('bla/1.2.276.0.7230010.3.1.4.8323329.644.1517'
    '875163.850974.png', 0)
img = pydicom.read_file('dicom-images-test/1.2.276.0.7230010.3.1.2.8323329.644.1517875163.850973/1.2.276.0.7230010.3.1.3.8323329.644.1517875163.850972/1.2.276.0.7230010.3.1.4.8323329.644.1517875163.850974.dcm').pixel_array  # noqa

model = FPNSegmentation('r50d')
model.load_state_dict(th.load('Both_SEG_logdir_104_f00/f00-ep-0016-val_dice-0.5837@0.20.pth'))  # noqa
model = model.cuda()
model.eval()

X = th.from_numpy(cv2.resize(img, (640, 640),
    interpolation=cv2.INTER_CUBIC)).unsqueeze(0).unsqueeze(0).cuda()
with th.no_grad():
  y_pred = model(X).cpu().numpy()
  y_pred_flip = th.flip(model(th.flip(X, (-1, ))), (-1, )).cpu().numpy()
  y_pred = 0.5 * (y_pred + y_pred_flip)
y_pred = (y_pred * 255).astype('uint8').squeeze()
plt.subplot(4, 1, 1)
plt.imshow(img)

plt.subplot(4, 1, 2)
plt.imshow(gold)

plt.subplot(4, 1, 3)
plt.title('MY PRED')
plt.imshow(y_pred)

plt.subplot(4, 1, 4)
plt.title('DIFF')
plt.imshow(y_pred != gold)

plt.show()
