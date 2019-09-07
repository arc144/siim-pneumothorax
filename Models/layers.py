import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


###########################################################################
########################## LAYER BLOCKS ###################################
###########################################################################

def Norm2d(planes):
    return nn.BatchNorm2d(planes)
    # return nn.GroupNorm(32, planes)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LinearReLuBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearReLuBn, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class PixelShuffle_ICNR(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, blur=False):
        super(PixelShuffle_ICNR, self).__init__()
        self.blur = blur
        self.conv = nn.Conv2d(in_channels,
                              out_channels * (scale ** 2),
                              kernel_size=1, padding=0)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur_kernel = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur_kernel(self.pad(x)) if self.blur else x


###########################################################################
########################### CONV BLOCKS ###################################
###########################################################################

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1, act=False):
        super(ConvBn2d, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True,
                              groups=groups,
                              dilation=dilation)
        self.bn = Norm2d(out_channels)
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class LargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(LargeKernelConv, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv1_1 = ConvBn2d(in_channels, out_channels,
                                kernel_size=(1, kernel_size),
                                padding=(0, pad))
        self.conv1_2 = ConvBn2d(out_channels, out_channels,
                                kernel_size=(kernel_size, 1),
                                padding=(pad, 0))

        self.conv2_1 = ConvBn2d(in_channels, out_channels,
                                kernel_size=(kernel_size, 1),
                                padding=(pad, 0))
        self.conv2_2 = ConvBn2d(out_channels, out_channels,
                                kernel_size=(1, kernel_size),
                                padding=(0, pad))

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        return x1 + x2


###########################################################################
####################### SQUEEZE EXCITATION BLOCKS #########################
###########################################################################

class SpatialGate2d(nn.Module):

    def __init__(self, in_channels):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.conv1(x)
        cal = self.sigmoid(cal)
        return cal * x


class ChannelGate2d(nn.Module):

    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.avg_pool(x)
        cal = self.fc1(cal)
        cal = self.relu(cal)
        cal = self.fc2(cal)
        cal = self.sigmoid(cal)

        return cal * x


class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate2d(channels)
        self.channel_gate = ChannelGate2d(channels, reduction=reduction)

    def forward(self, x, z=None):
        XsSE = self.spatial_gate(x)
        XcSe = self.channel_gate(x)
        return XsSE + XcSe


###########################################################################
########################## PYRAMID POOLING BLOCKS #########################
###########################################################################

class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_list, in_channels, size=(128, 128), mode='bilinear'):
        super(PyramidPoolingModule, self).__init__()
        self.pool_list = pool_list
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0) for _ in range(len(pool_list))])
        if mode == 'bilinear':
            self.upsample = nn.Upsample(size=size, mode=mode, align_corners=True)
        else:
            self.upsample = nn.Upsample(size=size, mode=mode)
            # self.conv2 = nn.Conv2d(in_channels * (1 + len(pool_list)), in_channels, kernel_size=1)

    def forward(self, x):
        cat = [x]
        for (k, s), conv in zip(self.pool_list, self.conv1):
            out = F.avg_pool2d(x, kernel_size=k, stride=s)
            out = conv(out)
            out = self.upsample(out)
            cat.append(out)
        out = torch.cat(cat, 1)
        # out = self.conv2(out)
        # out = self.relu(out)
        return out


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, depth, atrous_rates=[12, 24, 36]):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.conv1 = ConvBn2d(in_channels, depth, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = ConvBn2d(in_channels, depth, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0])
        self.conv3_2 = ConvBn2d(in_channels, depth, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1])
        self.conv3_3 = ConvBn2d(in_channels, depth, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.im_conv1 = ConvBn2d(in_channels, depth, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.final_conv1 = ConvBn2d(depth * 5, depth, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        rate1 = self.conv1(x)
        rate2 = self.conv3_1(x)
        rate3 = self.conv3_2(x)
        rate4 = self.conv3_3(x)

        im_level = self.pool(x)
        im_level = self.im_conv1(im_level)
        im_level = self.upsample(im_level)

        out = torch.cat([
            rate1,
            rate2,
            rate3,
            rate4,
            im_level
        ], 1)

        out = self.final_conv1(out)
        return out


###########################################################################
########################### ATTENTION BLOCKS ##############################
###########################################################################

class AttentionGate(nn.Module):
    def __init__(self, in_skip, in_g):
        super(AttentionGate, self).__init__()
        inter_c = in_skip // 2
        self.conv_skip = nn.Conv2d(in_skip, inter_c, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_g, inter_c, kernel_size=1, padding=0, bias=True)
        self.conv_psi = nn.Conv2d(inter_c, 1, kernel_size=1, padding=0, bias=True)
        self.W = nn.Sequential(nn.Conv2d(in_skip, in_skip, kernel_size=1, padding=0), Norm2d(in_skip))
        self.relu = nn.ReLU(inplace=True)
        # Initialise weights
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, x, g):
        theta_x = self.conv_skip(x)
        phi_g = self.conv_g(g)
        i = self.relu(theta_x + phi_g)
        i = self.conv_psi(i)
        i = torch.sigmoid(i)
        i = F.upsample(i, scale_factor=2, mode='bilinear', align_corners=False)
        i = i.expand_as(x) * x
        return self.W(i)


###########################################################################
############################ PANet BLOCKS #################################
###########################################################################

class FeaturePyramidAttention(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(FeaturePyramidAttention, self).__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1p = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv3a = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.conv7a = nn.Conv2d(channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv7b = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mode='std'):
        H, W = x.shape[2:]
        # down-path
        if mode == 'std':
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == 'reduced':
            xup1 = self.conv7a(x)
        elif mode == 'extended':
            xup1 = F.avg_pool2d(x, kernel_size=4, stride=4)
            xup1 = self.conv7a(xup1)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode='bilinear', align_corners=True)

        # Merge
        if mode == 'std':
            x1 = self.upsample(xup1) * x1
        elif mode == 'reduced':
            x1 = xup1 * x1
        elif mode == 'extended':
            x1 = F.upsample(xup1, scale_factor=4, mode='bilinear', align_corners=True) * x1
        x1 = x1 + gp
        return x1


class FeaturePyramidAttention_v2(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)
        self.conv1p = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)

        self.conv3a = ConvBn2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True)
        self.conv3b = ConvBn2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True)

        self.conv5a = ConvBn2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True)
        self.conv5b = ConvBn2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True)

        self.conv7a = ConvBn2d(channels, out_channels, kernel_size=7, stride=1, padding=3, act=True)
        self.conv7b = ConvBn2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, act=True)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mode='std'):
        H, W = x.shape[2:]
        # down-path
        if mode == 'std':
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == 'reduced':
            xup1 = self.conv7a(x)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode='bilinear', align_corners=True)

        # Merge
        if mode == 'std':
            x1 = self.upsample(xup1) * x1
        elif mode == 'reduced':
            x1 = xup1 * x1
        x1 = x1 + gp
        return x1


class GlobalAttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, out_channels=None):
        super(GlobalAttentionUpsample, self).__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)
        # GlobalPool and conv1
        cal1 = self.GPool(x)
        cal1 = self.conv1(cal1)
        cal1 = self.relu(cal1)

        # Calibrate skip connection
        skip = cal1 * skip
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x


class AttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, n_classes, out_channels=None):
        super().__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.rconv1 = nn.Conv2d(channels, n_classes, kernel_size=1, padding=0)
        self.gconv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)

        # GlobalPool and conv1
        gcalib = self.GPool(x)
        gcalib = self.gconv1(gcalib)
        gcalib = torch.sigmoid(gcalib)

        # RegionalPool
        rcalib = self.rconv1(x)
        rcalib = torch.sigmoid(rcalib)

        # Calibrate skip connection
        skip = (gcalib * skip) + (rcalib * skip)
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x, rcalib


###########################################################################
########################### DECODER BLOCKS ################################
###########################################################################

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels,
                 convT_channels=0, convT_ratio=2,
                 SE=False, residual=True):
        super(Decoder, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.SE = SE
        self.residual = residual
        self.convT_ratio = convT_ratio

        self.conv1 = ConvBn2d(in_channels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)

        if convT_ratio:
            self.convT = nn.ConvTranspose2d(convT_channels,
                                            convT_channels // convT_ratio,
                                            kernel_size=2, stride=2)
            if residual:
                self.conv_res = nn.Conv2d(convT_channels // convT_ratio,
                                          out_channels,
                                          kernel_size=1, padding=0)
        else:
            if residual:
                self.conv_res = nn.Conv2d(convT_channels, out_channels,
                                          kernel_size=1, padding=0)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

    def forward(self, x, skip=None, up_size=None):
        if self.convT_ratio:
            x = self.convT(x)
            x = self.activation(x)
        else:
            if up_size is not None:
                x = F.interpolate(x, size=(up_size, up_size), mode='bilinear',
                                  align_corners=True)  # False
            else:
                x = F.interpolate(x, scale_factor=2, mode='bilinear',
                                  align_corners=True)  # False

        residual = x
        if skip is not None:
            x = torch.cat([x, skip], 1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.SE:
            x = self.scSE(x)

        if self.residual:
            x += self.conv_res(residual)
        x = self.activation(x)
        return x


class PSDecoder(nn.Module):
    def __init__(self, in_channels, skip_chanels, channels, out_channels, PS=True, SE=False):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)

        if PS:
            self.upsample = nn.PixelShuffle(upscale_factor=2)
            factor = 4
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            factor = 1

        self.SE = SE

        self.conv1 = ConvBn2d(in_channels // factor + skip_chanels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels // factor, out_channels,
                                  kernel_size=1, padding=0)

        if SE:
            self.scSE = scSqueezeExcitationGate(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)

        residual = x
        if skip is not None:
            x = torch.cat([x, skip], 1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.SE:
            x = self.scSE(x)

        x += self.conv_res(residual)
        x = self.activation(x)
        return x


class AdjascentPrediction(nn.Module):

    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad2d(padding=1)
        self._range = range(3)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_x = self.pad(x)
        out = []
        for i in self._range:
            for j in self._range:
                out.append(pad_x[:, :, i:i + H, j: j + W])
        out = torch.cat(out, dim=1).mean(dim=1, keepdim=True)
        return out


###########################################################################
############################## SCNN BLOCKS ################################
###########################################################################

class ResidualBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = Norm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class GatedConv2d(nn.Module):

    def __init__(self, s_in, r_in):
        super().__init__()
        self.conv1_att = nn.Sequential(nn.Conv2d(r_in + s_in, s_in, kernel_size=1),
                                       Norm2d(s_in),
                                       nn.Sigmoid())
        self.conv1_weight = nn.Conv2d(s_in, s_in, kernel_size=1)

    def forward(self, s, r):
        # Compute alpha
        alpha = torch.cat([s, r], dim=1)
        alpha = self.conv1_att(alpha)

        # Apply residual and attention
        s = (s * alpha) + s

        # Final conv
        return self.conv1_weight(s)


class ShapeStream(nn.Module):

    def __init__(self, l1_in, l3_in, l4_in, l5_in):
        super().__init__()
        self.conv1_l1 = ConvBn2d(l1_in, l1_in, kernel_size=1, padding=0, act=True)
        self.conv1_l3 = ConvBn2d(l3_in, l1_in, kernel_size=1, padding=0, act=True)
        self.conv1_l4 = ConvBn2d(l4_in, l1_in, kernel_size=1, padding=0, act=True)
        self.conv1_l5 = ConvBn2d(l5_in, l1_in, kernel_size=1, padding=0, act=True)

        self.res_block1 = ResidualBlock(l1_in, l1_in)
        self.res_block2 = ResidualBlock(l1_in, l1_in)
        self.res_block3 = ResidualBlock(l1_in, l1_in)

        self.gconv1 = GatedConv2d(l1_in, l1_in)
        self.gconv2 = GatedConv2d(l1_in, l1_in)
        self.gconv3 = GatedConv2d(l1_in, l1_in)

        self.conv1_out = ConvBn2d(l1_in, 1, kernel_size=1, padding=0)
        self.conv1_grad_out = ConvBn2d(2, 1, kernel_size=1, padding=0)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, l1, l3, l4, l5, grad=None):
        x = self.conv1_l1(l1)
        x = self.res_block1(x)
        x = self.gconv1(x, self.up2(self.conv1_l3(l3)))

        x = self.res_block2(x)
        x = self.gconv2(x, self.up4(self.conv1_l4(l4)))

        x = self.res_block3(x)
        x = self.gconv3(x, self.up8(self.conv1_l5(l5)))

        out = self.conv1_out(x)

        if grad is not None:
            out_grad = self.conv1_grad_out(torch.cat([out, grad], dim=1))
        else:
            out_grad = out

        return out, out_grad


class EMAModule(nn.Module):

    def __init__(self, channels, K, lbda=1, alpha=0.1, T=3):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.lbda = lbda
        self.conv1_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.bn_out = nn.BatchNorm2d(channels)
        self.register_buffer('bases', torch.empty(K, channels)) # K x C
        # self.bases = Parameter(torch.empty(K, channels), requires_grad=False) # K x C
        nn.init.kaiming_uniform_(self.bases, a=math.sqrt(5))
        # self.bases.data = F.normalize(self.bases.data, dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.conv1_in(x).view(B, C, -1).transpose(1, -1) # B x N x C

        bases = self.bases[None, ...]
        x_in = x.detach()
        for i in range(self.T):
            # Expectation
            if i == (self.T - 1):
                x_in = x
            z = torch.softmax(self.lbda * torch.matmul(x_in, bases.transpose(1, -1)), dim=-1)  # B x N x K
            # Maximization
            bases = torch.matmul(z.transpose(1, 2), x_in) / (z.sum(1)[..., None] + 1e-12) # B x K x C
            bases = F.normalize(bases, dim=-1)
        if self.training:
            self.bases.data = (1 - self.alpha) * self.bases + self.alpha * bases.detach().mean(0)

        x = torch.matmul(z, bases).transpose(1, -1).view(B, C, H, W)
        x = self.conv1_out(x)
        x = self.bn_out(x)
        x += residual
        return x





