import torch.optim as optim
import torch
import torch.nn as nn

from Core.optmizers import AdamW, RAdam
from Core import metrics

import gc
import math
import os
import cv2
import datetime
import time
from tqdm import tqdm
import tensorboardX as tbX
from zipfile import ZipFile

import numpy as np
import pandas as pd
import apex

from Utils import TTA
from Utils import data_helpers as DH
from Utils import net_helpers as H
from Utils import postprocessing as PP
from Core import scheduler as sch

import matplotlib.pyplot as plt


###########################################################################
###############################  NETS #####################################
###########################################################################

class Segmentation():
    '''Base model for Networks.
    Subclass it and override methods for task specific metrics
    and funcionalities'''

    ######################### INIT FUNS #############################
    def __init__(self, net, mode='train', criterion=None, fold=None, debug=False, val_mode='max', comment=''):
        super().__init__()
        self.fold = fold
        self.debug = debug
        self.scheduler = None
        self.best_model_path = None
        self.epoch = 0
        self.val_mode = val_mode
        self.comment = comment
        self.is_training = False
        self.criterion = criterion
        self.tta = [TTA.Nothing(mode='sigmoid')]
        self.net = net
        self.freeze_encoder = False
        self.freeze_bn = False

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        self.train_log = {}
        self.val_log = {}
        if mode == 'train':
            self.create_save_folder()
            self.writer = tbX.SummaryWriter(log_dir=self.save_dir)

    # TODO: Implment a lr_finder method.

    def create_optmizer(self, optimizer='SGD', lr=1e-3, scheduler=None, gamma=0.25, patience=4,
                        milestones=None, T_max=10, T_mul=2, lr_min=0, freeze_encoder=False,
                        freeze_bn=False):
        self.lr = lr
        self.freeze_bn = freeze_bn
        self.net.cuda()
        self.set_encoder_trainable(not freeze_encoder)
        parameters = []
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            # elif name.startswith('encoder'):
            #     parameters.append({"params": param, "lr": 0.1 * self.lr})
            # param.requires_grad = False
            else:
                if name.endswith('weight'):
                    parameters.append({"params": param})
                else:
                    parameters.append({"params": param, "weight_decay": 0})

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(parameters,
                                       lr=self.lr, momentum=0.9, weight_decay=0.0001)

        elif optimizer == 'Adam':
            # self.optimizer = optim.Adam(parameters, lr=self.lr)
            self.optimizer = AdamW(parameters,
                                   lr=self.lr, weight_decay=5e-4)

        elif optimizer == 'RAdam':
            self.optimizer = RAdam(parameters,
                                   lr=self.lr)  # , weight_decay=5e-4)

        if scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='max',
                                                                  factor=gamma,
                                                                  patience=patience,
                                                                  verbose=True,
                                                                  threshold=0.01,
                                                                  min_lr=1e-05,
                                                                  eps=1e-08)

        elif scheduler == 'Milestones':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=milestones,
                                                            gamma=gamma,
                                                            last_epoch=-1)

        elif scheduler == 'CosineAnneling':
            self.scheduler = sch.CosineAnnealingLR(self.optimizer,
                                                   T_max=T_max,
                                                   T_mul=T_mul,
                                                   lr_min=lr_min,
                                                   val_mode=self.val_mode,
                                                   last_epoch=-1,
                                                   save_snapshots=True,
                                                   save_all=False)
        elif scheduler == 'OneCycleLR':
            self.scheduler = sch.OneCycleLR(self.optimizer,
                                            num_steps=T_max,
                                            lr_range=(lr_min, lr))

        elif scheduler == 'Exponential':
            exp_decay = math.exp(-0.01)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=exp_decay)

        self.net, self.optimizer = apex.amp.initialize(self.net, self.optimizer, opt_level='O1')

    ######################### TRAINING #############################
    def train_network(self, train_loader, val_loader,
                      grad_acc=1, n_epoch=10, epoch_per_val=3,
                      mixup=0):
        with H.timer('Training Fold {}'.format(self.fold)):
            # Required to correct behavior when resuming training
            n_epoch = n_epoch + self.epoch
            print('Model created, total of {} parameters'.format(
                sum(p.numel() for p in self.net.parameters())))

            # self.do_validation(val_loader)
            while self.epoch < n_epoch:
                self.epoch += 1
                lr = np.mean([param_group['lr'] for param_group in self.optimizer.param_groups])
                with H.timer('Train Epoch {:}/{:} - LR: {:.4E}'.format(self.epoch, n_epoch, lr)):
                    # Training step
                    self.training_step(train_loader, grad_acc=grad_acc, mixup=mixup)
                    #  Validation
                    if not (self.epoch - 1) % epoch_per_val:
                        # Prepare gallery for validation and next example mining
                        self.do_validation(val_loader)
                    # Learning Rate Scheduler
                    if self.scheduler is not None:
                        if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                            self.scheduler.step(self.val_log['total_mAP'][-1])
                        elif type(self.scheduler).__name__ == 'CosineAnnealingLR':
                            self.best_model_path = self.scheduler.step(
                                self.epoch,
                                save_dict=dict(dice=self.val_log['dice'][-1],
                                               ptx_dice=self.val_log['ptx_dice'][-1],
                                               save_dir=self.save_dir,
                                               fold=self.fold,
                                               state_dict=self.net.state_dict()))
                        else:
                            self.scheduler.step()
                    # Save best model
                    if type(self.scheduler).__name__ != 'CosineAnnealingLR':
                        self.save_best_model(self.val_log['dice'][-1])

                self.update_tbX(self.epoch)
            # self.save_training_log()
            self.writer.close()

    def training_step(self, train_loader, grad_acc=1,
                      # ths=0.5, noise_th=0,
                      ths=0.2, noise_th=75.0 * (384 / 128.0) ** 2,
                      mixup=0):
        '''Training step of a single epoch'''
        self.set_mode('train')
        # Define the frequency metrics are computed
        n_iter = len(train_loader.batch_sampler.sampler) / train_loader.batch_sampler.batch_size

        # Begin epoch loop
        loss_list = []
        dice_list = []
        ptx_dice_list = []
        acc_list = []
        pbar = tqdm(enumerate(train_loader), total=n_iter)
        self.optimizer.zero_grad()
        for i, (im_id, meta, im, mask) in pbar:
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(train_loader.batch_size, 1, figsize=(20, 40))
            # for i in range(train_loader.batch_size):
            #     axs[i].imshow(im.numpy()[i, 0, ...].squeeze(), cmap=plt.cm.bone)
            #     axs[i].imshow(mask.numpy()[i, ...].squeeze(), alpha=0.3)
            # plt.show()

            im = im.cuda()
            mask = mask.cuda()
            # Mixup?
            if mixup > 0:
                im, mask = DH.mixup_data(im, mask, alpha=mixup)
            # Forward propagation
            logit = self.net(im)
            loss = self.net.loss(self.criterion, logit, mask).mean()
            loss_list.append(loss.item())

            with apex.amp.scale_loss(loss / grad_acc, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            # (loss / grad_acc).backward()
            # torch.nn.utils.clip_grad_value_(self.net.parameters(), 1)

            pred = torch.sigmoid(logit[0]).data.cpu().numpy()
            # pred = torch.softmax(logit, 1)[:, 1:, ...].data.cpu().numpy()
            pred[pred.reshape(pred.shape[0], -1).sum(-1) < noise_th, ...] = 0.0
            pred = (pred > ths)

            dice, ptx_dice = metrics.cmp_dice(pred, mask.data.cpu().numpy())
            dice = dice.mean()
            ptx_dice = ptx_dice.mean()
            acc = metrics.cmp_cls_acc(pred, mask.data.cpu().numpy()).mean()
            dice_list.append(dice)
            ptx_dice_list.append(ptx_dice)
            acc_list.append(acc)

            pbar.set_postfix_str('loss: {:.3f} dice: {:.3f}, ptx_dice: {:.3f}, acc: {:.3f}'.format(loss.item(), dice,
                                                                                                   ptx_dice, acc))

            if (i + 1) % grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Append epoch data to metrics dict
        out = dict(loss=loss_list, dice=dice_list)

        for metric, value in out.items():
            H.update_log(self.train_log, metric, np.mean(value))

    def do_validation(self, val_loader,
                      # ths=0.5, noise_th=0,
                      ths=0.2, noise_th=75.0 * (384 / 128.0) ** 2,
                      ):
        '''Validation step after epoch end'''
        self.set_mode('valid')
        self.net.cuda()

        loader = enumerate(val_loader)

        loss_list, dice_list, ptx_dice_list, acc_list = [], [], [], []
        for i, (im_ids, meta, im, mask) in loader:
            n = len(im_ids)
            if n < 1:
                continue

            im = im.cuda()
            mask = mask.cuda()

            with torch.no_grad():
                logit = self.net(im)
                pred = torch.sigmoid(logit[0]).cpu().numpy()
                loss = self.net.loss(self.criterion, logit, mask).cpu().numpy()

            pred[pred.reshape(pred.shape[0], -1).sum(-1) < noise_th, ...] = 0.0
            pred = (pred > ths)
            dice, ptx_dice = metrics.cmp_dice(pred, mask.cpu().numpy())
            acc = metrics.cmp_cls_acc(pred, mask.cpu().numpy())

            loss_list.extend(np.atleast_1d(loss))
            dice_list.extend(dice)
            acc_list.extend(acc)
            ptx_dice_list.extend(ptx_dice)

            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(val_loader.batch_size, 3, figsize=(20, 40))
            # for i in range(val_loader.batch_size):
            #     axs[i, 0].imshow(im.numpy()[i, ...].squeeze(), cmap=plt.cm.bone)
            #     axs[i, 0].imshow(mask.numpy()[i, ...].squeeze(), alpha=0.3)
            #     axs[i, 1].imshow(torch.sigmoid(logit).cpu().numpy()[i, ...].squeeze())
            #     axs[i, 2].imshow(np.sum(
            #         [(j + 1) * (255 / len(pred[i])) * pred[i][j] / 255 for j in range(len(pred[i]))], axis=0).squeeze())
            # plt.show()

        metrics_out = dict(dice=np.mean(dice_list),
                           ptx_dice=np.mean(ptx_dice_list),
                           acc=np.mean(acc_list),
                           loss=np.mean(loss_list))

        # Append epoch data to metrics dict
        for metric, value in metrics_out.items():
            H.update_log(self.val_log, metric, value)

        self.print_metrics()
        return metrics_out

    ######################## HELPER FUNS ##########################

    def predict(self,
                test_loader,
                TTA,
                raw=False,
                pred_zip=None,
                tgt_size=1024,
                ths=0.2, noise_th=75.0 * (384 / 128.0) ** 2,
                pbar=False):
        self.set_mode('valid')
        self.net.cuda()

        n_images = len(test_loader.dataset)
        im_size = test_loader.dataset.tgt_size
        n_iter = n_images / test_loader.batch_size
        if pbar:
            loader = tqdm(enumerate(test_loader), total=n_iter)
        else:
            loader = enumerate(test_loader)

        index_vec = []
        meta_vec = []

        count = 0
        for i, (im_ids, meta, im, mask) in loader:
            n = len(im_ids)
            if n < 1:
                continue
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(test_loader.batch_size // 4, 4)
            # axs = axs.ravel()
            # for i in range(len(index)):
            #     axs[i].imshow(np.moveaxis(im.numpy()[i, ...], 0, -1))
            # plt.show()

            with torch.no_grad():
                for j, tta in enumerate(TTA):
                    if not j:
                        pred = tta(self.net, im)
                    else:
                        pred = (j * pred + tta(self.net, im)) / (j + 1)

            if not raw:
                pred[pred.reshape(pred.shape[0], -1).sum(-1) < noise_th, ...] = 0.0
                # pred_cls = (pred > 0.75)
                pred = (pred > ths) * 255
                # pred[pred_cls.reshape(pred.shape[0], -1).sum(1) == 0] = 0.0
                pred = np.array([
                    cv2.resize(pred[j, 0].astype(np.uint8),
                               (tgt_size, tgt_size)) for j in range(pred.shape[0])])
                pred = (pred > 127) * 255
            else:
                pred = np.array([
                    cv2.resize(pred[j, 0],
                               (tgt_size, tgt_size)) for j in range(pred.shape[0])])

            if pred_zip is not None:
                for z, im_id in enumerate(im_ids):
                    pred_fn = im_id + '.png'
                    yp_img = np.uint8(pred[z] * 255)
                    img_bytes = cv2.imencode('.png', yp_img)[1].tobytes()
                    with ZipFile(pred_zip, 'a') as f:
                        f.writestr(pred_fn, img_bytes)

            if not i:
                pred_vec = np.zeros((n_images, *pred.shape[-2:]), dtype=np.float16)
                # mask_vec = None
                mask_vec = np.zeros((n_images, *mask.shape[-2:]), dtype=np.float16)

            pred_vec[count: count + n] = pred if len(pred.shape) == 3 else pred.squeeze(1)
            mask_vec[count: count + n] = mask if len(mask.shape) == 3 else mask.squeeze(1)
            index_vec.extend(im_ids)
            meta_vec.extend(meta)

            count += n
            gc.collect()

        # mask_vec = np.concatenate(mask_vec, axis=0)
        # pred_vec = np.asarray(pred_vec)
        return index_vec, meta_vec, pred_vec, mask_vec

    def print_metrics(self):
        msg = ''
        for metric, value in self.train_log.items():
            msg += 'train {:}: {:.3f}\t'.format(metric, value[-1])
        for metric, value in self.val_log.items():
            msg += 'val {:}: {:.3f}\t'.format(metric, value[-1])
        print(msg)

    def set_encoder_trainable(self, state):
        parameters = self.net.encoder.parameters()
        # parameters = self.net[0].parameters()
        if state:
            for param in parameters:
                param.requires_grad = True
        else:
            for param in parameters:
                param.requires_grad = False

        self.freeze_encoder = not state

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.net.eval()
            self.is_training = False
        elif mode in ['train']:
            self.is_training = True
            self.net.train()
            if self.freeze_encoder:
                self.net.encoder.apply(H.set_batchnorm_eval)
            if self.freeze_bn:
                self.net.apply(H.set_batchnorm_eval)
        else:
            raise NotImplementedError

    ####################### I/O FUNS ##############################
    def update_tbX(self, step):
        '''Update SummaryWriter for tensorboard'''
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('lr_group{}'.format(i),
                                   param_group['lr'],
                                   step)
        for tk, value in self.train_log.items():
            self.writer.add_scalar('train_{}'.format(tk), value[-1], step)
        for vk, value in self.val_log.items():
            self.writer.add_scalar('val_{}'.format(vk), value[-1], step)

    def save_best_model(self, metric):
        if (self.val_mode == 'max' and metric > self.best_metric) or (
                self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            if self.fold is not None:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    '{:}_Fold{:}_Epoch{}_val{:.3f}'.format(date, self.fold, self.epoch, metric))
            else:
                self.best_model_path = os.path.join(
                    self.save_dir,
                    '{:}_Epoch{}_val{:.3f}'.format(date, self.epoch, metric))

            torch.save(self.net.state_dict(), self.best_model_path)

    def save_training_log(self):
        d = dict()
        for tk in self.train_log.keys():
            d['train_{}'.format(tk)] = self.train_log[tk]
        for vk in self.val_log.keys():
            d['val_{}'.format(vk)] = self.val_log[vk]

        df = pd.DataFrame(d)
        df.index += 1
        df.index.name = 'Epoch'

        date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
        if self.fold is not None:
            p = os.path.join(
                self.save_dir,
                '{:}_Fold{:}_TrainLog.csv'.format(date, self.fold))
        else:
            p = os.path.join(
                self.save_dir,
                '{:}_TrainLog.csv'.format(date))

        df.to_csv(p, sep=";")

        with open(p, 'a') as fd:
            fd.write(self.comment)

    def load_model(self, path=None, best_model=False):
        if best_model:
            self.net.load_state_dict(torch.load(self.best_model_path))
        else:
            sd = torch.load(path)
            try:
                state_dict = sd['state_dict']
            except KeyError:
                state_dict = sd

            # del state_dict['loss.weight'] #TODO: REMOVE THIS

            self.net.load_state_dict(state_dict, strict=False)
            # self.net.load_state_dict(torch.load(path)['state_dict'], strict=True)
        print('Model checkpoint loaded from: {}'.format(path))

    def create_save_folder(self):
        cwd = os.getcwd()
        name = type(self.net).__name__
        fold = 'Fold{}'.format(self.fold)
        exp_id = time.strftime('%d%h_%H:%M')
        self.save_dir = os.path.join(cwd, 'Saves', name, fold, exp_id)
        # Create dirs recursively if does not exist
        _dir = ''
        for d in self.save_dir.split('/'):
            if d == '':
                continue
            _dir += '/' + d
            if not os.path.exists(_dir):
                os.makedirs(_dir)
