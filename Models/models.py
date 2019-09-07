import torch
import torch.nn as nn
import torch.nn.functional as F
import Models.layers as L
from Backbone import *
from torch.nn import init as init
import torchvision
from efficientnet_pytorch import EfficientNet
import numpy as np

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class UNetDilatedResNet34H(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = dilated_resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)
        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.center = nn.Sequential(
            L.ConvBn2d(512, 256,
                       kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            L.ConvBn2d(256, 128,
                       kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = L.Decoder(128 + 128, 256, 64, convT_channels=64, convT_ratio=0, SE=False, residual=False)
        self.decoder3 = L.Decoder(64 + 64, 128, 64, convT_channels=64, convT_ratio=0, SE=False, residual=False)
        self.decoder2 = L.Decoder(64 + 64, 64, 64, convT_channels=64, convT_ratio=0, SE=False, residual=False)

        self.logit = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=3, stride=2, padding=1)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/16

        f = self.center(e4)  # 1/16

        d4 = self.decoder4(f, e2)  # 1/8
        d3 = self.decoder3(d4, e1)  # 1/4
        d2 = self.decoder2(d3, e0)  # 1/2

        hc = torch.cat([
            d2,
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False),
        ], 1)

        logit = F.interpolate(self.logit(hc),
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class PANetResNet34(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 256
        self.encoder2 = self.encoder.layer2  # 512
        self.encoder3 = self.encoder.layer3  # 1024
        self.encoder4 = self.encoder.layer4  # 2048

        self.center_conv = nn.Sequential(
            L.ConvBn2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(64)

        self.decoder5 = L.GlobalAttentionUpsample(256, 64)
        self.decoder4 = L.GlobalAttentionUpsample(128, 64)
        self.decoder3 = L.GlobalAttentionUpsample(64, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=3, stride=2, padding=1)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        e4 = self.center_conv(e4)
        f = self.FPA(e4, mode='std')

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class PANetResNet50(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet50(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.center_conv = nn.Sequential(
            L.ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention_v2(512, 64)

        self.decoder5 = L.GlobalAttentionUpsample(1024, 64)
        self.decoder4 = L.GlobalAttentionUpsample(512, 64)
        self.decoder3 = L.GlobalAttentionUpsample(256, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        # if H < 512:
        #     mode = 'reduced'
        # else:
        #     mode = 'std'

        f = self.center_conv(e4)  # 1/32
        f = self.FPA(f, mode='std')

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        # s = F.dropout2d(d, p=0.40, training=self.is_training)

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class PANetDilatedResNet34(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = dilated_resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.center_conv = nn.Sequential(
            L.ConvBn2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(64)

        self.decoder5 = L.GlobalAttentionUpsample(256, 64)
        self.decoder4 = L.GlobalAttentionUpsample(128, 64)
        self.decoder3 = L.GlobalAttentionUpsample(64, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/16

        # if H < 512:
        #     mode = 'reduced'
        # else:
        #     mode = 'std'

        f = self.center_conv(e4)  # 1/16
        f = self.FPA(f, mode='std')

        d5 = self.decoder5(f, e3, up=False)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        # s = F.dropout2d(d, p=0.40, training=self.is_training)

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class PANetFishNet150(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = fishnet150(pretrained=pretrained)

        self.center_conv = nn.Sequential(
            L.ConvBn2d(2112, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(512, 64)

        self.decoder5 = L.GlobalAttentionUpsample(1600, 64)
        self.decoder4 = L.GlobalAttentionUpsample(832, 64)
        self.decoder3 = L.GlobalAttentionUpsample(320, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        all_feat = self.encoder(x)
        e1 = all_feat[7]  # 1/4
        e2 = all_feat[8]  # 1/8
        e3 = all_feat[9]  # 1/16
        e4 = all_feat[10]  # 1/32

        if H < 512:
            mode = 'reduced'
        else:
            mode = 'std'

        f = self.center_conv(e4)  # 1/32
        f = self.FPA(f, mode=mode)

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class PANetFishNet99(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = fishnet99(pretrained=pretrained)

        self.center_conv = nn.Sequential(
            L.ConvBn2d(2112, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(64)

        self.decoder5 = L.GlobalAttentionUpsample(1600, 64)
        self.decoder4 = L.GlobalAttentionUpsample(832, 64)
        self.decoder3 = L.GlobalAttentionUpsample(320, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        all_feat = self.encoder(x)
        e1 = all_feat[7]  # 1/4
        e2 = all_feat[8]  # 1/8
        e3 = all_feat[9]  # 1/16
        e4 = all_feat[10]  # 1/32

        if H < 512:
            mode = 'reduced'
        else:
            mode = 'std'

        f = self.center_conv(e4)  # 1/32
        f = self.FPA(f, mode=mode)

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class AttPANetDilatedResNet34(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = dilated_resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.center_conv = nn.Sequential(
            L.ConvBn2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.cls_conv = L.ConvBn2d(64, 64, kernel_size=3, padding=1)

        self.FPA = L.FeaturePyramidAttention(64)

        self.decoder5 = L.GlobalAttentionUpsample(256, 64)
        self.decoder4 = L.GlobalAttentionUpsample(128, 64)
        self.decoder3 = L.GlobalAttentionUpsample(64, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.seg_conv = L.ConvBn2d(64, 64, kernel_size=3, padding=1)
        self.seg_logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.cls_logit = nn.Linear(64, 1)

        self.logit = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/16

        # if H < 512:
        #     mode = 'reduced'
        # else:
        #     mode = 'std'

        f = self.center_conv(e4)  # 1/16
        cls_maps = self.cls_conv(f)
        f = self.FPA(f, mode='std')

        d5 = self.decoder5(f, e3, up=False)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4
        seg_maps = self.seg_conv(d3)

        cls_logit = self.cls_logit(F.adaptive_avg_pool2d(cls_maps, 1).view(batch_size, -1))
        seg_logit = self.upsample(self.seg_logit(seg_maps))

        fuse = torch.cat([seg_maps,
                          self.upsample(cls_maps)],
                         dim=1)

        logit = self.upsample(self.logit(fuse))
        return (logit, seg_logit, cls_logit)

    def loss(self, criterion, logit, mask):
        all_logit, seg_logit, cls_logit = logit
        all_loss, seg_loss, cls_loss = 0, 0, 0

        B = mask.shape[0]
        cls_label = (mask.view(B, -1).sum(1) > 0)

        if cls_label.sum() > 0:
            seg_loss = criterion(seg_logit[cls_label], mask[cls_label]).mean() / 10
        cls_loss = F.binary_cross_entropy_with_logits(cls_logit,
                                                      cls_label.unsqueeze(1).float()).mean() / 40
        all_loss = criterion(all_logit, mask).mean()
        return all_loss + seg_loss + cls_loss


class HRNetResNetW48(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        cfg = dict()
        cfg['FINAL_CONV_KERNEL'] = 1
        cfg['STAGE2'] = dict(
            NUM_MODULES=1,
            NUM_BRANCHES=2,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4),
            NUM_CHANNELS=(48, 96),
            FUSE_METHOD='SUM')
        cfg['STAGE3'] = dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(48, 96, 192),
            FUSE_METHOD='SUM')
        cfg['STAGE4'] = dict(
            NUM_MODULES=3,
            NUM_BRANCHES=4,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4, 4),
            NUM_CHANNELS=(48, 96, 192, 384),
            FUSE_METHOD='SUM')
        if pretrained:
            cfg['PRETRAINED'] = '/media/hdd/Kaggle/Pneumothorax/Data/Weights/hrnet_w48_pascal_context_cls60_480x480.pth'
        else:
            cfg['PRETRAINED'] = ''
        self.encoder = get_seg_model(cfg)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
        return [self.upsample(self.encoder(x))]

    def loss(self, criterion, logits, mask):
        return criterion(logits[0], mask)


class GatedSResNet34(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = resnet34(pretrained=pretrained)

        self.encoder1 = nn.Sequential(self.encoder.conv1,
                                      self.encoder.bn1,
                                      self.encoder.relu)

        self.encoder2 = self.encoder.layer1  # 64
        self.encoder3 = self.encoder.layer2  # 128
        self.encoder4 = self.encoder.layer3  # 256
        self.encoder5 = self.encoder.layer4  # 512

        self.shape_stream = L.ShapeStream(64, 128, 256, 512)

        self.fpa = L.FeaturePyramidAttention_v2(512, 1)

        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.fusion = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x, grad=None):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e1 = self.encoder1(x)  # 1/2
        e1 = F.max_pool2d(e1, kernel_size=3, stride=2, padding=1)  # 1/4

        e2 = self.encoder1(e1)  # 1/4
        e3 = self.encoder2(e2)  # 1/8
        e4 = self.encoder3(e3)  # 1/16
        e5 = self.encoder4(e4)  # 1/32
        r_out = self.up8(self.fpa(e5))

        s_out, s_grad_out = self.shape_stream(e1, e3, e4, e5, grad)

        # Fuse
        out = self.fusion(torch.cat([r_out, s_grad_out], dim=1))
        return [out, s_out]

    def loss(self, criterion, logit, mask):
        # mask 0 - mask, 1 - boundary
        pass


class EMANetDilatedResNet34_v2(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = dilated_resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.ema = nn.Sequential(L.EMAModule(512, 64, lbda=1, alpha=0.1, T=3),
                                 L.ConvBn2d(512, 64, kernel_size=3, padding=1, act=True))

        self.decoder5 = L.GlobalAttentionUpsample(256, 64)
        self.decoder4 = L.GlobalAttentionUpsample(128, 64)
        self.decoder3 = L.GlobalAttentionUpsample(64, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/16

        f = self.ema(e4)  # 1 /16

        d5 = self.decoder5(f, e3, up=False)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class EMANetResNet101_v2(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet101(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 256
        self.encoder2 = self.encoder.layer2  # 512
        self.encoder3 = self.encoder.layer3  # 1024
        self.encoder4 = self.encoder.layer4  # 2048

        self.ema = nn.Sequential(L.ConvBn2d(2048, 512, kernel_size=3, padding=1, act=True),
                                 L.EMAModule(512, 64, lbda=1, alpha=0.1, T=3),
                                 L.ConvBn2d(512, 64, kernel_size=3, padding=1, act=True))

        self.decoder5 = L.GlobalAttentionUpsample(1024, 64)
        self.decoder4 = L.GlobalAttentionUpsample(512, 64)
        self.decoder3 = L.GlobalAttentionUpsample(256, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        f = self.ema(e4)  # 1 /32

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class EMANetSEResNet50_v2(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = se_resnet50(pretrained=pretrained)

        self.conv1 = self.encoder.layer0

        self.encoder1 = self.encoder.layer1  # 256
        self.encoder2 = self.encoder.layer2  # 512
        self.encoder3 = self.encoder.layer3  # 1024
        self.encoder4 = self.encoder.layer4  # 2048

        self.ema = nn.Sequential(L.ConvBn2d(2048, 512, kernel_size=3, padding=1, act=True),
                                 L.EMAModule(512, 64, lbda=1, alpha=0.1, T=3),
                                 L.ConvBn2d(512, 64, kernel_size=3, padding=1, act=True))

        self.decoder5 = L.GlobalAttentionUpsample(1024, 64)
        self.decoder4 = L.GlobalAttentionUpsample(512, 64)
        self.decoder3 = L.GlobalAttentionUpsample(256, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        e1 = self.encoder1(e0)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        f = self.ema(e4)  # 1 /32

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)


class RANetR34(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.ema = nn.Sequential(L.EMAModule(512, 64, lbda=1, alpha=0.1, T=3),
                                 L.ConvBn2d(512, 64, kernel_size=3, padding=1, act=True))

        self.decoder5 = L.AttentionUpsample(256, 64, 1)
        self.decoder4 = L.AttentionUpsample(128, 64, 1)
        self.decoder3 = L.AttentionUpsample(64, 64, 1)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        f = self.ema(e4)  # 1 /32

        d5, out5 = self.decoder5(f, e3, up=True)  # 1/16
        d4, out4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3, out3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit, out3, out4, out5]

    def loss(self, criterion, logit, mask):
        mask_s4 = F.avg_pool2d(mask, 4)
        mask_s8 = F.avg_pool2d(mask_s4, 2)
        mask_s16 = F.avg_pool2d(mask_s8, 2)

        l1 = criterion(logit[0], mask).mean()
        l4 = criterion(logit[1], mask_s4).mean()
        l8 = criterion(logit[2], mask_s8).mean()
        l16 = criterion(logit[3], mask_s16).mean()

        return (l1 * 0.7) + (0.1 * l4) + (0.1 * l8) + (0.1 * l16)


class EffiNetB5(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.encoder(x)
        return [x]

    def loss(self, criterion, logit, mask):
        label = (mask.view(mask.shape[0], -1).sum(1) > 0)[:, None].float()
        return criterion(logit[0], label)


class R34(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet34(pretrained=pretrained)
        self.fc = nn.Linear(512, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return [x]

    def loss(self, criterion, logit, mask):
        label = (mask.view(mask.shape[0], -1).sum(1) > 0)[:, None].float()
        return criterion(logit[0], label)


class R50(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512
        self.fc = nn.Linear(2048, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x)
        x = self.encoder.maxpool(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return [x]

    def loss(self, criterion, logit, mask):
        label = (mask.view(mask.shape[0], -1).sum(1) > 0)[:, None].float()
        return criterion(logit[0], label)

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, **kwargs):
        super(DenseNet121, self).__init__()
        self.encoder = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14))

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
        x = self.encoder(x)
        return [x[:, 7:8]]

    def loss(self, criterion, logit, mask):
        label = (mask.view(mask.shape[0], -1).sum(1) > 0)[:, None].float()
        return criterion(logit[0], label)

class PANetDenseNet121(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = densenet121(pretrained=pretrained)

        self.center_conv = nn.Sequential(
            L.ConvBn2d(1024, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(128)

        self.decoder5 = L.GlobalAttentionUpsample(1024, 128)
        self.decoder4 = L.GlobalAttentionUpsample(512, 128)
        self.decoder3 = L.GlobalAttentionUpsample(256, 128)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.encoder.features[:4](x)  # 1/2
        e1 = self.encoder.features[4:5](e0)  # 1/4
        e2 = self.encoder.features[5:7](e1)  # 1/8
        e3 = self.encoder.features[7:9](e2)  # 1/16
        e4 = self.encoder.features[9:](e3)  # 1/32

        e4 = self.center_conv(e4)
        f = self.FPA(e4, mode='std')

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)

def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)

class PANetEffNetB5(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)

        self.center_conv = nn.Sequential(
            L.ConvBn2d(512, 40, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(40)

        self.decoder5 = L.GlobalAttentionUpsample(176, 40)
        self.decoder4 = L.GlobalAttentionUpsample(64, 40)
        self.decoder3 = L.GlobalAttentionUpsample(40, 40)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(40, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        # Stem
        x = relu_fn(self.encoder._bn0(self.encoder._conv_stem(x)))
        e0 = x # 48 1/2
        # Blocks
        # 3 8 13 27
        for idx, block in enumerate(self.encoder._blocks):
            drop_connect_rate = self.encoder._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.encoder._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == 2:
                e1 = x # 24 1/2
            elif idx == 7:
                e2 = x # 40 1/4
            elif idx == 12:
                e3 = x # 64 1/8
            elif idx == 26:
                e4 = x # 176 1/16

        f = self.center_conv(x)
        f = self.FPA(f, mode='std')

        d5 = self.decoder5(f, e4, up=True)  # 1/16
        d4 = self.decoder4(d5, e3, up=True)  # 1/8
        d3 = self.decoder3(d4, e2, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return [logit]

    def loss(self, criterion, logit, mask):
        return criterion(logit[0], mask)
