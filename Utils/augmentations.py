import random

import cv2
import imgaug.augmenters as iaa
import imgaug.parameters as iap
import numpy as np
import six.moves as sm
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage
from torchvision.models.resnet import resnet34

class RotateCropResize(iaa.meta.Augmenter):
    '''Class that allows performing rotations without introducing artifacts.
       This class is also extended to be used in conjunction with ImgAug.
       So far it only works with square images'''

    def __init__(self, rotate=(-15, 15), name=None, deterministic=False, random_state=None):
        super(RotateCropResize, self).__init__(name=name,
                                               deterministic=deterministic,
                                               random_state=random_state)

        self.rotate = iap.handle_continuous_param(rotate, "rotate",
                                                  value_range=None,
                                                  tuple_to_uniform=True,
                                                  list_to_choice=True)

    def get_parameters(self):
        return [self.rotate]

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.rotate.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            images[i] = self.rotate_crop_resize(images[i], samples[i])
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError

    @staticmethod
    def rotate_crop_resize(im, angle):
        '''Rotate image and crop black artifacts'''
        H, W = im.shape[:2]
        # Expects square images
        assert H == W, 'Image is not square'
        x = ndimage.rotate(im, angle)
        edge = int(np.ceil(H * np.sin(abs(angle) * np.pi / 180)))
        x = x[edge:-(edge + 1), edge:-(edge + 1), ...]
        x = cv2.resize(x, (H, W))
        return x


class CLAHE(iaa.meta.Augmenter):

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8),
                 name=None, deterministic=False, random_state=None):
        super(CLAHE, self).__init__(name=name,
                                    deterministic=deterministic,
                                    random_state=random_state)

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def get_parameters(self):
        return [self.clip_limit, self.tile_grid_size]

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        for i in sm.xrange(nb_images):
            images[i] = self._clahe(images[i], self.clip_limit, self.tile_grid_size)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        raise NotImplementedError

    @staticmethod
    def _clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        if img.dtype != np.uint8:
            raise TypeError('clahe supports only uint8 inputs')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return img


####################################################################################
################################ AUG SCHEMES #######################################
####################################################################################
Aug1 = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # iaa.Flipud(0.2),  # vertically flip 20% of all images
        # iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            shear=(-10, 10),
            # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10),
            order=1,  # use bilinear interpolation (fast)
            cval=0,  # if mode is constant, use a cval between 0 and 255
            mode='constant'
        )),
        iaa.OneOf([
            iaa.GammaContrast((0.5, 1.5)),
            iaa.LinearContrast((0.5, 1.5)),
            iaa.ContrastNormalization((0.70, 1.30)),
        ])
    ])

Aug2a = iaa.Sequential([
    iaa.Fliplr(0.5),
    # iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=3), iaa.MotionBlur(k=3)])),
    iaa.Add((-5, 5), per_channel=0.5),
    iaa.Multiply((0.9, 1.1), per_channel=0.5),
    iaa.Sometimes(0.5, iaa.Affine(
        scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
        translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        shear=(-5, 5),
        rotate=(-30, 30)
    )),
], random_order=True)

Aug2b = iaa.Sequential([
    iaa.Fliplr(0.5),
    # iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=3), iaa.MotionBlur(k=3)])),
    # iaa.Add((-5, 5), per_channel=0.5),
    # iaa.Multiply((0.9, 1.1), per_channel=0.5),
    iaa.Sometimes(0.5, iaa.Affine(
        scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
        translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        shear=(-5, 5),
        rotate=(-30, 30)
    )),
], random_order=True)

Aug2d = iaa.Sequential([
    iaa.Fliplr(0.5),
    # iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=3), iaa.MotionBlur(k=3)])),
    iaa.OneOf([
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.Multiply((0.9, 1.1), per_channel=0.5)]),
    iaa.Sometimes(0.5, iaa.Affine(
        scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
        translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        shear=(-5, 5),
        rotate=(-30, 30)
    )),
], random_order=True)

Aug2e = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Add((-5, 5), per_channel=0.5),
    iaa.Multiply((0.9, 1.1), per_channel=0.5),
    # iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=3), iaa.MotionBlur(k=3)])),
    iaa.Sometimes(0.5, iaa.Affine(
        scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        shear=(-10, 10),
        rotate=(-30, 30)
    )),
    iaa.OneOf([
        iaa.GammaContrast((0.5, 1.5)),
        iaa.LinearContrast((0.5, 1.5)),
        iaa.ContrastNormalization((0.70, 1.30)),
    ])
], random_order=True)

Aug3 = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5, iaa.OneOf([iaa.AverageBlur(k=(3, 5)), iaa.MotionBlur(k=(3, 5))])),
    iaa.Add((-15, 15), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.5),
    iaa.Sometimes(0.5, iaa.Affine(
        scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
        translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},
        shear=(-15, 15),
        rotate=(-30, 30)
    )),
]
    , random_order=True)


class AutoAugment(object):
    def __init__(self):
        self.policies = [
            ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
            ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
            ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
            ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
            ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
            ['Color', 0.4, 3, 'Brightness', 0.6, 7],
            ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
            ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
            ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
            ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
            ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
            ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
            ['Brightness', 0.9, 6, 'Color', 0.2, 8],
            ['Solarize', 0.5, 2, 'Invert', 0, 0.3],
            ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
            ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
            ['Color', 0.9, 9, 'Equalize', 0.6, 6],
            ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
            ['Brightness', 0.1, 3, 'Color', 0.7, 0],
            ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
            ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
            ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
            ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
            ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
        ]

    def __call__(self, img):
        img = Image.fromarray(img)
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def shear_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1,
                                  img.shape[1] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150 / 331, 150 / 331, 11)

    transform_matrix = np.array(
        [[1, 0, img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])],
         [0, 1, 0],
         [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def rotate(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
        img[:, :, c],
        affine_matrix,
        offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img


def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img


def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))))
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1]))
    return img


def cutout(org_img, magnitude=None):
    img = np.array(org_img)

    magnitudes = np.linspace(0, 60 / 331, 11)

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(img.shape[0] * random.uniform(magnitudes[magnitude], magnitudes[magnitude + 1])))
    top = np.random.randint(0 - mask_size // 2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size // 2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    img = Image.fromarray(img)

    return img


class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        img = np.array(img)

        mask_val = img.mean()

        top = np.random.randint(0 - self.length // 2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length // 2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length

        top = 0 if top < 0 else top
        left = 0 if left < 0 else top

        img[top:bottom, left:right, :] = mask_val

        img = Image.fromarray(img)

        return img
