"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific
   heads on top of it.

See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object
  Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified
  Architecture for Instance and Semantic Segmentation

"""
from torchvision import models
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
try:
    import timm
except ImportError as e:
    print(e)
    print("Install timm via: `pip install --upgrade timm`")
    import gluon_resnet as timm


def convert_to_inplace_relu(model):
    # make all relus inplace: https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7  # noqa
  for child_name, child in model.named_children():
    if isinstance(child, nn.ReLU):
      setattr(model, child_name, nn.ReLU(inplace=True))
    else:
      convert_to_inplace_relu(child)


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=3, alpha=0.9):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.alpha = alpha

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(x_t, mu)      # b * n * k
                z = F.softmax(z, dim=2)     # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       # b * c * k
                mu = self._l2norm(mu, dim=1)

            if self.training:
                self.mu = self.alpha * self.mu + \
                    (1. - self.alpha) * mu.mean(dim=0, keepdim=True)

        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        x = x.view(b, c, h, w)              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class FPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral
       connections. Can be used as feature extractor for object detection
       or segmentation.
    """

    def __init__(self, slug, num_filters=256, pretrained=True):
        """Creates an `FPN` instance for feature extraction.

        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number fo input channels
        """

        super().__init__()
        assert pretrained

        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r50d':
            self.resnet = timm.create_model('gluon_resnet50_v1d',
                pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        elif slug == 'r101d':
            self.resnet = timm.create_model('gluon_resnet101_v1d',
                pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048

        else:
            assert False, "Bad slug: %s" % slug

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.lateral4 = Conv1x1(num_bottleneck_filters, num_filters)
        self.lateral3 = Conv1x1(num_bottleneck_filters // 2, num_filters)
        self.lateral2 = Conv1x1(num_bottleneck_filters // 4, num_filters)
        self.lateral1 = Conv1x1(num_bottleneck_filters // 8, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

    def forward_s4(self, enc0):
        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.interpolate(map4, scale_factor=2,
            mode="nearest")
        map2 = lateral2 + nn.functional.interpolate(map3, scale_factor=2,
            mode="nearest")
        map1 = lateral1 + nn.functional.interpolate(map2, scale_factor=2,
            mode="nearest")
        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        map1, map2, map3, map4 = self.forward_s4(enc0)
        return enc0, map1, map2, map3, map4


class FPNSegmentation(nn.Module):
    """Semantic segmentation model on top of a Feature Pyramid Network (FPN).
    """

    def __init__(self, slug, num_classes=1, num_filters=128,
            num_filters_fpn=256, pretrained=True, num_input_channels=1,
            output_size=(1024, 1024), ema=False):
        """Creates an `FPNSegmentation` instance for feature extraction.

        Args:
          slug: model slug e.g. 'r18', 'r101' for ResNet
          num_classes: number of classes to predict
          num_filters: the number of filters in each segmentation head pyramid
                       level
          num_filters_fpn: the number of filters in each FPN output pyramid
                           level
          pretrained: use ImageNet pre-trained backbone feature extractor
          num_input_channels: number of input channels e.g. 3 for RGB
          output_size. Tuple[int, int] height, width
        """

        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        self.first_down = nn.Conv2d(num_input_channels, 3, kernel_size=3,
            padding=1, stride=2, bias=False)
        self.fpn = FPN(slug=slug, num_filters=num_filters_fpn,
                pretrained=pretrained)
        # The segmentation heads on top of the FPN

        self.head1 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head2 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head3 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))
        self.head4 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters),
            Conv3x3(num_filters, num_filters))

        if not ema:
            self.final = nn.Conv2d(4 * num_filters, num_classes, kernel_size=3,
                padding=1)
        else:
            print("Using EMA module")
            self.final = nn.Sequential(
                EMAU(4 * num_filters, 64),
                nn.Conv2d(4 * num_filters, num_classes, kernel_size=3,
                padding=1))

        self.up8 = torch.nn.Upsample(scale_factor=8, mode='nearest')
        self.up4 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.sigmoid = nn.Sigmoid()
        self.output_size = output_size

    def forward(self, x):
        # normalize
        x = x / 127.5 - 1.0
        x = self.first_down(x)
        enc0, map1, map2, map3, map4 = self.fpn(x)

        h4 = self.head4(map4)
        h3 = self.head3(map3)
        h2 = self.head2(map2)
        h1 = self.head1(map1)

        map4 = self.up8(h4)
        map3 = self.up4(h3)
        map2 = self.up2(h2)
        map1 = h1

        final_map = torch.cat([map4, map3, map2, map1], 1)
        final = self.final(final_map)

        return self.sigmoid(nn.functional.interpolate(final,
            scale_factor=8, mode="bilinear", align_corners=False))


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    model = FPNSegmentation('r34', ema=False)
    X = torch.randn(1, 1, 1024, 1024)
    out = model(X)
    print(out.shape)
