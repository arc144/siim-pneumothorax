import os
from datetime import datetime

import numpy as np
import pandas as pd

from Core.dataset import DatasetFactory
from Core.tasks import Segmentation
from functools import reduce
from operator import or_


def get_model_for_inference(cfg):
    # LOAD NET
    print('#####################################')
    print('NET: {}'.format(cfg.model.backbone.__name__))
    print('TGT_SIZE: {}'.format(cfg.image.tgt_size))
    print('COLOR: {}\n'.format(cfg.image.color))
    print('BATCH_SIZE: {}'.format(cfg.image.batch_size))
    print('TTAs: {}'.format([type(x).__name__ for x in cfg.inference.TTA]))
    print('#####################################')

    net = cfg.model.architecture(cfg.model.backbone, cfg.model.emb_dim,
                                 cfg.model.n_classes, cfg.model.pool,
                                 cfg.loss.fun,
                                 dict(s=cfg.loss.s, margin=cfg.loss.margin),
                                 pretrained=cfg.model.pretrained)

    model = Segmentation(net, margin=cfg.loss.margin,
                         mode='test', debug=False,
                         fold=cfg.fold)

    if cfg.model.weights is not None:
        model.load_model(cfg.model.weights)

    return model


def extract_features(model, dataloader, TTA, save_path=None):
    ids, feat, cls = model.inference(dataloader,
                                     TTA=TTA,
                                     tta_mode='cat',
                                     pbar=True)

    if save_path is not None:
        np.save(save_path + '_ids', ids)
        np.save(save_path + '_feat', feat)
        np.save(save_path + '_cls', cls)

    return ids, feat, cls


def sample_gal_from_landmark_cls(gal_csv, landmark_df, threshold):
    gal_df = pd.read_csv(gal_csv)
    # assert np.all(np.isin(gal_df.id.values, landmark_df.id.values)), \
    #     "'Landmark_csv' must contain all gallery's ids!"
    ix = landmark_df.loc[landmark_df['prob0'] > threshold, 'id'].values
    gal_df = gal_df.set_index('id').loc[ix].reset_index()
    return gal_df


def sample_from_gallery(tocmp_path, max_im_per_cls, precmp_cls=None):
    to_cmp_df = pd.read_csv(tocmp_path)
    if precmp_cls is None:
        return to_cmp_df.groupby('class').apply(
            lambda x: x.sample(min(len(x), max_im_per_cls), replace=False))

    def get_n_im_per_cls(x):
        cls = x['class'].values[0]
        curr_n = precmp_count[np.where(precmp_unique == cls)[0]]

        n_im = max(0, max_im_per_cls - curr_n)
        return min(n_im, len(x))

    precmp_unique, precmp_count = np.unique(precmp_cls, return_counts=True)
    print('Got {} features already computed.'.format(len(precmp_cls)))
    return to_cmp_df.groupby('class').apply(
        lambda x: x.sample(get_n_im_per_cls(x), replace=False))


def extract_gal_features(model, gal_path, cfg, filter_df=None):
    date = datetime.strftime(datetime.now(), '_%d-%m-%H:%M')
    if filter_df is not None:
        gal_df = sample_gal_from_landmark_cls(gal_path,
                                              filter_df,
                                              cfg.inference.landmark_cls_threshold)
        # restart from where it was left off
        # gal_df = gal_df.iloc[1097133:]
        print('Now extracting {} features of {} classes'.format(
            len(gal_df), len(gal_df['class'].unique())))
        gal_loader = DatasetFactory.from_df(gal_df, cfg).yield_loader(is_test=True)
    else:
        gal_loader = DatasetFactory(gal_path, cfg).yield_loader(is_test=True)

    gal_save_path = os.path.join(cfg.inference.feat_path,
                                 '{}_gal'.format(date))
    gal_ids, gal_feat, gal_cls = extract_features(model,
                                                  gal_loader,
                                                  cfg.inference.TTA,
                                                  gal_save_path)
    return gal_ids, gal_feat, gal_cls


def extract_query_features(model, query_path, cfg):
    date = datetime.strftime(datetime.now(), '_%d-%m-%H:%M')
    query_loader = DatasetFactory(query_path, cfg).yield_loader(is_test=True)
    query_save_path = os.path.join(cfg.inference.feat_path,
                                   '{}_query'.format(date))
    query_ids, query_feat, _ = extract_features(model,
                                                query_loader,
                                                cfg.inference.TTA,
                                                query_save_path)
    return query_ids, query_feat

def load_precmp_feats(path, filter_df=None, threshold=0.99):
    ids = np.load(path + '_ids.npy')
    feat = np.load(path + '_feat.npy')
    cls = np.load(path + '_cls.npy')

    if filter_df is not None:
        mask = filter_df.set_index('id').loc[ids, 'prob0'].values > threshold
        print('{} ids out of {}'.format(mask.sum(), len(mask)))
        ids = ids[mask]
        feat = feat[mask]
        cls = cls[mask]
    return ids, feat, cls
