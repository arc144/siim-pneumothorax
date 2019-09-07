import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from skimage import exposure
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler
import Utils.data_helpers as H


###########################################################################
###############################  DATASET ##################################
###########################################################################


class PneumoDataset(Dataset):
    """This is a class that inherits the Dataset class from PyTorch."""

    def __init__(self, df, is_train=False, transform=None, tgt_size=None):
        self.df = df.set_index('ImageId')

        if is_train:
            if 'EncodedPixels' in self.df.columns:
                self.gb = self.df.groupby('ImageId')
                self.df['count'] = self.gb.transform('count')['EncodedPixels']
                self.df.loc[self.df['EncodedPixels'] == '-1', 'count'] = 0
            else:
                self.df['count'] = self.df['Pneumothorax'].values

        self.tgt_size = tgt_size
        self.transform = transform
        self.ids = np.unique(self.df.index.values)
        self.size = len(self.ids)
        self.mapping = dict(zip(self.ids, range(self.size)))
        print('Dataset comprised of {} images'.format(self.size))

    def __len__(self):
        return self.size

    def apply_transform(self, im, mask):
        if self.transform is not None:
            mask = SegmentationMapOnImage(mask.astype(bool), im.shape)
            im, mask = self.transform(image=im, segmentation_maps=mask)
            mask = mask.get_arr_int() * 255

        return im, mask

    def load_image_gt(self, index):
        '''Load image and gt from a subset dataframe'''
        image_id = self.ids[index]

        path = self.df.loc[image_id, 'Path']
        image, meta = H.read_image(path if isinstance(path, str) else path.values[0])
        h, w = image.shape[:2]

        if 'EncodedPixels' in self.df.columns:
            rles = self.df.loc[image_id, 'EncodedPixels']
            if isinstance(rles, str) and rles == '-1':
                mask = np.zeros((h, w))
            else:
                if isinstance(rles, pd.Series):
                    rles = list(rles.values)

                elif isinstance(rles, str):
                    rles = [rles]

                mask = [H.rle2mask(rle,
                                   meta['width'],
                                   meta['height']) for rle in rles]
                mask = np.max(mask, axis=0)
        elif 'Pneumothorax' in self.df.columns:
            mask = self.df.loc[image_id, 'Pneumothorax']
            mask = np.ones((h, w)) * 255 * mask
        else:
            mask = np.zeros((h, w))

        return image_id, meta, image, mask

    def __getitem__(self, index):

        # Load gts, images and process images
        image_id, meta, image, mask = self.load_image_gt(index)
        if image is None:
            return image_id, None, None, None

        image = H.resize_image(image, self.tgt_size)
        mask = H.resize_image(mask, self.tgt_size)
        mask = (mask > 127.5).astype(np.uint8) * 255

        image, mask = self.apply_transform(image, mask)
        # image = H.input_lbp(image)

        image = H.uint2float(image)
        image = H.toTensor(image)
        # if mask is not None:
        mask = H.uint2float(mask)
        mask = H.toTensor(mask)
        # else:
        #     mask = torch.from_numpy(np.zeros((1, 256, 256)))

        return image_id, meta, image, mask


class BalanceClassSampler(Sampler):

    def __init__(self, dataset, remove_bg_only=False):
        self.remove_bg_only = remove_bg_only
        self.dataset = dataset

        if self.remove_bg_only:
            self.length = len(self.dataset.df[self.dataset.df['count'] >= 1].index.values)
        else:
            self.length = len(self.dataset)

    def __iter__(self):
        if self.remove_bg_only:
            pos_ids = self.dataset.df[self.dataset.df['count'] >= 1].index.values
            pos_index = [self.dataset.mapping[id] for id in pos_ids]
            pos = np.random.choice(pos_index, len(pos_index), replace=False)
            return iter(pos)

        pos_ids = self.dataset.df[self.dataset.df['count'] >= 1].index.values
        pos_index = [self.dataset.mapping[id] for id in pos_ids]
        neg_ids = self.dataset.df[self.dataset.df['count'] == 0].index.values
        neg_index = [self.dataset.mapping[id] for id in neg_ids]

        half = self.length // 2 + 1
        pos = np.random.choice(pos_index, half, replace=True)
        neg = np.random.choice(neg_index, half, replace=True)

        l = np.stack([pos, neg]).T
        l = l.reshape(-1)
        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length


###########################################################################
########################### DATASET FACTORY ###############################
###########################################################################


class DatasetFactory():
    def __init__(self, csv_path, cfg):
        self._cfg = cfg
        self.load_dfs(csv_path)

    def load_dfs(self, csv_path):
        '''Load the CSVs from path'''
        self.df = pd.read_csv(csv_path)

    def yield_loader(self, is_test=False):
        '''
        Proceed with PyTorch data pipeline.
        Return the dataloaders used in training
        '''
        df = self.df

        if is_test:

            if 'Set' in self.df.columns:
                df = self.df[self.df['Set'] == 'val']

            dataset = PneumoDataset(df,
                                    tgt_size=self._cfg.image.tgt_size,
                                    is_train=False,
                                    )

            dataloader = DataLoader(dataset,
                                    shuffle=False,
                                    num_workers=self._cfg.image.workers,
                                    batch_size=self._cfg.image.batch_size,
                                    collate_fn=H.default_batch_collate,
                                    pin_memory=True)



        else:

            if 'Set' in self.df.columns:
                df = self.df[self.df['Set'] == 'train']

            dataset = PneumoDataset(df,
                                    transform=self._cfg.image.aug,
                                    tgt_size=self._cfg.image.tgt_size,
                                    is_train=True
                                    )

            dataloader = DataLoader(dataset,
                                    num_workers=self._cfg.image.workers,
                                    collate_fn=H.default_batch_collate,
                                    pin_memory=True,
                                    sampler=BalanceClassSampler(
                                        dataset,
                                        remove_bg_only=self._cfg.train.remove_bg_only),
                                    batch_size=self._cfg.image.batch_size,
                                    drop_last=True,
                                    worker_init_fn=self._cfg._init_fn)

        return dataloader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Config import UNet
    import os

    os.chdir('../')
    cfg = UNet.Config()
    dataset = DatasetFactory('/media/hdd/Kaggle/Pneumothorax/Data/Folds/fold0.csv', cfg)
    dataloader = dataset.yield_loader(is_test=True)
    for index, meta, im, mask in dataloader:
        fig, axis = plt.subplots(4, 4, figsize=(20, 20))
        axis = axis.ravel()
        for i in range(16):
            axis[i].imshow(np.moveaxis(im[i].numpy(), 0, -1))
            axis[i].imshow(np.moveaxis(mask[i].numpy(), 0, -1), alpha=0.3)
            # axis[i].set_title(label[i].item())
        break

    plt.show()
